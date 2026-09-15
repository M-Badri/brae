// Side-by-side correctness and steady-state kernel benchmark for the isolated factory copy.
// The production implementation comes from brae_core.  The copied implementation is included below
// with only its public entry points renamed, allowing both paths to consume the very same device data.
#include "primitive_mesh.cuh"
#include "device_ldu.cuh"
#include "device_sym_gauss_seidel.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define buildDeviceGaussSeidelLevels factoryBuildDeviceGaussSeidelLevels
#define gsSingleBlockWalk factoryGsSingleBlockWalk
#define gsLevelGatherEnabled factoryGsLevelGatherEnabled
#define gsLevelCoefsRefresh factoryGsLevelCoefsRefresh
#define gsLevelCellsRefresh factoryGsLevelCellsRefresh
#define gsLevelsFor factoryGsLevelsFor
#define deviceSymGaussSeidelSweepExact factoryDeviceSymGaussSeidelSweepExact
#define deviceSymGaussSeidelSweepExactFused factoryDeviceSymGaussSeidelSweepExactFused
#include "device_sym_gauss_seidel.cu"
#undef buildDeviceGaussSeidelLevels
#undef gsSingleBlockWalk
#undef gsLevelGatherEnabled
#undef gsLevelCoefsRefresh
#undef gsLevelCellsRefresh
#undef gsLevelsFor
#undef deviceSymGaussSeidelSweepExact
#undef deviceSymGaussSeidelSweepExactFused

using namespace brae;

namespace
{

struct HostSystem
{
    std::string name;
    int nCells = 0;
    std::vector<label> owner, nei, ownerStart, losort, losortStart;
    std::vector<scalar> diag, upper, lower, b, psi0;
};

void finishSystem(HostSystem& h)
{
    const int nF = static_cast<int>(h.nei.size());
    h.ownerStart.assign(static_cast<std::size_t>(h.nCells) + 1, 0);
    h.losortStart.assign(static_cast<std::size_t>(h.nCells) + 1, 0);
    h.losort.resize(static_cast<std::size_t>(nF));
    for (int f = 0; f < nF; ++f)
    {
        ++h.ownerStart[static_cast<std::size_t>(h.owner[f]) + 1];
        ++h.losortStart[static_cast<std::size_t>(h.nei[f]) + 1];
    }
    for (int c = 0; c < h.nCells; ++c)
    {
        h.ownerStart[static_cast<std::size_t>(c) + 1] += h.ownerStart[static_cast<std::size_t>(c)];
        h.losortStart[static_cast<std::size_t>(c) + 1] += h.losortStart[static_cast<std::size_t>(c)];
    }
    std::vector<label> cursor = h.losortStart;
    for (int f = 0; f < nF; ++f) h.losort[static_cast<std::size_t>(cursor[h.nei[f]]++)] = f;

    h.upper.resize(static_cast<std::size_t>(nF));
    h.lower.resize(static_cast<std::size_t>(nF));
    h.diag.assign(static_cast<std::size_t>(h.nCells), scalar(1.0));
    for (int f = 0; f < nF; ++f)
    {
        const scalar up = scalar(-0.035) - scalar(0.0005)*(f % 23);
        const scalar lo = scalar(-0.041) - scalar(0.0004)*(f % 19);
        h.upper[static_cast<std::size_t>(f)] = up;
        h.lower[static_cast<std::size_t>(f)] = lo;
        h.diag[static_cast<std::size_t>(h.owner[f])] += std::fabs(up);
        h.diag[static_cast<std::size_t>(h.nei[f])] += std::fabs(lo);
    }
    h.b.resize(static_cast<std::size_t>(h.nCells));
    h.psi0.resize(static_cast<std::size_t>(h.nCells));
    for (int c = 0; c < h.nCells; ++c)
    {
        h.b[static_cast<std::size_t>(c)] = scalar(0.7) + scalar(0.2)*std::sin(scalar(c)*scalar(0.013));
        h.psi0[static_cast<std::size_t>(c)] = scalar(0.1)*std::cos(scalar(c)*scalar(0.007));
    }
}

HostSystem readCase(const std::string& caseDir)
{
    PrimitiveMesh mesh;
    mesh.read(caseDir + "/constant/polyMesh");
    HostSystem h;
    h.name = caseDir;
    h.nCells = static_cast<int>(mesh.nCells());
    h.nei = mesh.neighbour();
    h.owner.assign(mesh.owner().begin(), mesh.owner().begin() + static_cast<std::ptrdiff_t>(h.nei.size()));
    finishSystem(h);
    return h;
}

HostSystem makeGrid(int nx, int ny)
{
    HostSystem h;
    h.name = "grid-" + std::to_string(nx) + "x" + std::to_string(ny);
    h.nCells = nx*ny;
    h.owner.reserve(static_cast<std::size_t>((nx - 1)*ny + (ny - 1)*nx));
    h.nei.reserve(h.owner.capacity());
    for (int y = 0; y < ny; ++y)
    {
        for (int x = 0; x < nx; ++x)
        {
            const label c = static_cast<label>(y*nx + x);
            if (x + 1 < nx) { h.owner.push_back(c); h.nei.push_back(c + 1); }
            if (y + 1 < ny) { h.owner.push_back(c); h.nei.push_back(c + nx); }
        }
    }
    finishSystem(h);
    return h;
}

struct Comparison
{
    bool bitwise = true;
    std::size_t differing = 0;
    scalar maxAbs = 0;
    scalar maxRel = 0;
};

Comparison compare(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    Comparison r;
    r.bitwise = a.size() == b.size()
             && std::memcmp(a.data(), b.data(), a.size()*sizeof(scalar)) == 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::memcmp(&a[i], &b[i], sizeof(scalar)) != 0) ++r.differing;
        const scalar d = std::fabs(a[i] - b[i]);
        r.maxAbs = std::max(r.maxAbs, d);
        r.maxRel = std::max(r.maxRel, d/std::max(std::fabs(a[i]), scalar(1e-300)));
    }
    return r;
}

float timedBatch(const std::function<void()>& reset,
                 const std::function<void()>& sweep,
                 int sweeps)
{
    reset();
    sweep();
    cudaCheck(cudaDeviceSynchronize(), "factory benchmark warm-up");
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaCheck(cudaEventCreate(&start), "factory benchmark event create");
    cudaCheck(cudaEventCreate(&stop), "factory benchmark event create");
    cudaCheck(cudaEventRecord(start), "factory benchmark event record");
    for (int i = 0; i < sweeps; ++i) sweep();
    cudaCheck(cudaEventRecord(stop), "factory benchmark event record");
    cudaCheck(cudaEventSynchronize(stop), "factory benchmark event sync");
    float ms = 0;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop), "factory benchmark elapsed time");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms/static_cast<float>(sweeps);
}

struct Timing
{
    float productionMs = 0;
    float factoryMs = 0;
};

Timing medianTiming(const std::function<void()>& resetProduction,
                    const std::function<void()>& sweepProduction,
                    const std::function<void()>& resetFactory,
                    const std::function<void()>& sweepFactory,
                    int sweeps)
{
    std::vector<float> prod, fact;
    for (int round = 0; round < 7; ++round)
    {
        if (round & 1)
        {
            fact.push_back(timedBatch(resetFactory, sweepFactory, sweeps));
            prod.push_back(timedBatch(resetProduction, sweepProduction, sweeps));
        }
        else
        {
            prod.push_back(timedBatch(resetProduction, sweepProduction, sweeps));
            fact.push_back(timedBatch(resetFactory, sweepFactory, sweeps));
        }
    }
    std::sort(prod.begin(), prod.end());
    std::sort(fact.begin(), fact.end());
    return {prod[prod.size()/2], fact[fact.size()/2]};
}

int run(const HostSystem& h)
{
    DeviceLduMatrix matrix = buildDeviceLdu(h.diag, h.upper, h.lower, h.owner, h.nei, h.nCells);
    const DeviceLduView A = matrix.view();
    DeviceBuffer<scalar> dB;
    dB.copyFrom(h.b);
    DeviceGaussSeidelLevels lv = buildDeviceGaussSeidelLevels(
        h.owner, h.nei, h.nCells, h.losort, h.losortStart, h.ownerStart);
    GSLevelCoefs lc;
    GSLevelCells cc;
    gsLevelCoefsRefresh(A, lv, lc);
    gsLevelCellsRefresh(A, dB, lv, cc);
    GSLevelCoefs factoryLc;
    GSLevelCells factoryCc;
    factoryGsLevelCoefsRefresh(A, lv, factoryLc);
    factoryGsLevelCellsRefresh(A, dB, lv, factoryCc);
    cudaCheck(cudaDeviceSynchronize(), "factory benchmark operand refresh");

    const int exactSweeps = 25;
    const int timingSweeps = std::max(50, std::min(400, 120000/std::max(1, lv.levels())));
    std::printf("\n%s: %d cells, %zu faces, %d levels, widest %d\n",
                h.name.c_str(), h.nCells, h.nei.size(), lv.levels(), lv.maxLevelWidth);

#include "refresh_benchmark.inc"

    DeviceBuffer<scalar> prodPsi, factPsi;
    prodPsi.copyFrom(h.psi0);
    factPsi.copyFrom(h.psi0);
    for (int i = 0; i < exactSweeps; ++i)
    {
        deviceSymGaussSeidelSweepExact(A, dB, prodPsi, lv, true, &lc, &cc);
        factoryDeviceSymGaussSeidelSweepExact(A, dB, factPsi, lv, true, &factoryLc, &factoryCc);
    }
    cudaCheck(cudaDeviceSynchronize(), "factory benchmark scalar identity");
    std::vector<scalar> prodH, factH;
    prodPsi.copyTo(prodH);
    factPsi.copyTo(factH);
    const Comparison scalarCmp = compare(prodH, factH);
    std::printf("  scalar identity after %d sweeps: %s; differing %zu, max abs %.3e, max rel %.3e\n",
                exactSweeps, scalarCmp.bitwise ? "BITWISE" : "DIFF", scalarCmp.differing,
                static_cast<double>(scalarCmp.maxAbs), static_cast<double>(scalarCmp.maxRel));

    const Timing scalarTime = medianTiming(
        [&] { prodPsi.copyFrom(h.psi0); },
        [&] { deviceSymGaussSeidelSweepExact(A, dB, prodPsi, lv, true, &lc, &cc); },
        [&] { factPsi.copyFrom(h.psi0); },
        [&] { factoryDeviceSymGaussSeidelSweepExact(A, dB, factPsi, lv, true, &factoryLc, &factoryCc); },
        timingSweeps);
    std::printf("  scalar median: production %.3f ms/sweep, factory %.3f ms/sweep, speedup %.3fx\n",
                scalarTime.productionMs, scalarTime.factoryMs,
                scalarTime.productionMs/scalarTime.factoryMs);

    constexpr int NC = 3;
    DeviceBuffer<scalar> diag[NC], rhs[NC], prodComp[NC], factComp[NC];
    GSLevelCells compCells[NC], factoryCompCells[NC];
    const GSLevelCells* compCellPtr[NC] = {&compCells[0], &compCells[1], &compCells[2]};
    const GSLevelCells* factoryCompCellPtr[NC] = {&factoryCompCells[0], &factoryCompCells[1], &factoryCompCells[2]};
    std::vector<scalar> diagH[NC], rhsH[NC], psiH[NC];
    GSFusedOperands prodOps, factOps;
    prodOps.nComp = factOps.nComp = NC;
    for (int k = 0; k < NC; ++k)
    {
        diagH[k] = h.diag;
        rhsH[k] = h.b;
        psiH[k] = h.psi0;
        for (int c = 0; c < h.nCells; ++c)
        {
            diagH[k][static_cast<std::size_t>(c)] *= scalar(1) + scalar(0.01*k);
            rhsH[k][static_cast<std::size_t>(c)] += scalar(0.03*k)*std::cos(scalar(c)*scalar(0.021));
            psiH[k][static_cast<std::size_t>(c)] += scalar(0.02*k);
        }
        diag[k].copyFrom(diagH[k]);
        rhs[k].copyFrom(rhsH[k]);
        prodComp[k].copyFrom(psiH[k]);
        factComp[k].copyFrom(psiH[k]);
        DeviceLduView Ak = A;
        Ak.diag = diag[k].data();
        gsLevelCellsRefresh(Ak, rhs[k], lv, compCells[k]);
        factoryGsLevelCellsRefresh(Ak, rhs[k], lv, factoryCompCells[k]);
        prodOps.diag[k] = factOps.diag[k] = diag[k].data();
        prodOps.b[k] = factOps.b[k] = rhs[k].data();
        prodOps.psi[k] = prodComp[k].data();
        factOps.psi[k] = factComp[k].data();
    }
#include "fused_refresh_benchmark.inc"

    for (int i = 0; i < exactSweeps; ++i)
    {
        deviceSymGaussSeidelSweepExactFused(A, prodOps, lv, true, &lc, compCellPtr);
        factoryDeviceSymGaussSeidelSweepExactFused(A, factOps, lv, true, &factoryLc, factoryCompCellPtr);
    }
    cudaCheck(cudaDeviceSynchronize(), "factory benchmark fused identity");
    bool fusedBitwise = true;
    scalar fusedAbs = 0, fusedRel = 0;
    std::size_t fusedDiffering = 0;
    for (int k = 0; k < NC; ++k)
    {
        prodComp[k].copyTo(prodH);
        factComp[k].copyTo(factH);
        const Comparison c = compare(prodH, factH);
        fusedBitwise = fusedBitwise && c.bitwise;
        fusedDiffering += c.differing;
        fusedAbs = std::max(fusedAbs, c.maxAbs);
        fusedRel = std::max(fusedRel, c.maxRel);
    }
    std::printf("  fused-3 identity after %d sweeps: %s; differing %zu, max abs %.3e, max rel %.3e\n",
                exactSweeps, fusedBitwise ? "BITWISE" : "DIFF", fusedDiffering,
                static_cast<double>(fusedAbs), static_cast<double>(fusedRel));

    auto resetProdFused = [&]
    {
        for (int k = 0; k < NC; ++k)
        {
            prodComp[k].copyFrom(psiH[k]);
            prodOps.psi[k] = prodComp[k].data();
        }
    };
    auto resetFactFused = [&]
    {
        for (int k = 0; k < NC; ++k)
        {
            factComp[k].copyFrom(psiH[k]);
            factOps.psi[k] = factComp[k].data();
        }
    };
    const Timing fusedTime = medianTiming(
        resetProdFused,
        [&] { deviceSymGaussSeidelSweepExactFused(A, prodOps, lv, true, &lc, compCellPtr); },
        resetFactFused,
        [&] { factoryDeviceSymGaussSeidelSweepExactFused(A, factOps, lv, true, &factoryLc, factoryCompCellPtr); },
        timingSweeps);
    std::printf("  fused-3 median: production %.3f ms/sweep, factory %.3f ms/sweep, speedup %.3fx\n",
                fusedTime.productionMs, fusedTime.factoryMs,
                fusedTime.productionMs/fusedTime.factoryMs);

    return refreshCmp.bitwise && scalarCmp.bitwise && fusedBitwise ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 4 && std::string(argv[1]) == "--grid")
            return run(makeGrid(std::stoi(argv[2]), std::stoi(argv[3])));
        const std::string caseDir = argc > 1 ? argv[1] : "validation/T3A";
        return run(readCase(caseDir));
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "factory benchmark failed: %s\n", e.what());
        return 2;
    }
}
