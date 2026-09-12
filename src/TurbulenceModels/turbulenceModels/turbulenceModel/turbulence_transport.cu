#include "turbulence_transport.cuh"
#include <algorithm>   // std::max
#include "device_mesh.cuh"     // deviceDivUpwindCoeffs / deviceDivLimitedCoeffs / deviceLaplacian*
#include "device_kepsilon.cuh" // deviceGaussGrad, deviceCellLimitGrad, deviceBCValue
#include "device_blas.cuh"     // deviceAxpy
#include "device_fvoptions.cuh"   // deviceSetValues: fvMatrix::setValues
#include "device_pcg.cuh"
#include "device_amg.cuh"      // deviceSymGaussSeidel, when the case names a smoothSolver
#include "device_simple.cuh"    // deviceRelaxDiag -- fvMatrix::relax
#include <cstdio>
#include <string>
#include <cmath>

namespace brae {
namespace gpu {
namespace turbulence {

namespace {

// FP-2: M -= the laplacian coefficients, five arrays of three lengths in one launch. Each statement is
// axpyKernel's `y += a * x` with a = -1, so nvcc emits the same fused multiply-add and the bits match the
// five separate launches this replaces (held by the SST dumps and written fields on aerofoilNACA0012 and
// the residual lines on squareBend and injectorPipe, all bit-identical before and after).
__global__ void subtractLaplacianKernel(
    int nC, int nF, int nB,
    const scalar* __restrict__ lDiag, const scalar* __restrict__ lUp, const scalar* __restrict__ lLo,
    const scalar* __restrict__ lIC, const scalar* __restrict__ lBC,
    scalar* __restrict__ diag, scalar* __restrict__ upper, scalar* __restrict__ lower,
    scalar* __restrict__ iC, scalar* __restrict__ bC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const scalar a = scalar(-1.0);
    if (i < nC) diag[i] += a * lDiag[i];
    if (i < nF) { upper[i] += a * lUp[i]; lower[i] += a * lLo[i]; }
    if (i < nB) { iC[i] += a * lIC[i]; bC[i] += a * lBC[i]; }
}

void subtractLaplacian(
    const DeviceBuffer<scalar>& lDiag, const DeviceBuffer<scalar>& lUp, const DeviceBuffer<scalar>& lLo,
    const DeviceBuffer<scalar>& lIC, const DeviceBuffer<scalar>& lBC, PressureMatrix& M)
{
    const int nC = static_cast<int>(lDiag.size()), nF = static_cast<int>(lUp.size()), nB = static_cast<int>(lIC.size());
    const int n = std::max(nC, std::max(nF, nB));
    if (n <= 0) return;
    constexpr int tpb = 256;
    subtractLaplacianKernel<<<(n + tpb - 1) / tpb, tpb>>>(
        nC, nF, nB, lDiag.data(), lUp.data(), lLo.data(), lIC.data(), lBC.data(),
        M.diag.data(), M.upper.data(), M.lower.data(), M.iC.data(), M.bC.data());
    cudaCheck(cudaGetLastError(), "subtractLaplacian");
}
// Each device module in the tree carries its own copy of this two-liner (rhoEEqn.cu, rhoPEqn.cu,
// rhoPcEqn.cu, kEpsilon.cu). Kept local here for the same reason rather than adding a public symbol
// for a memset.
void zeroed(DeviceBuffer<scalar>& b, int n)
{
    b.resize(static_cast<std::size_t>(n));
    cudaCheck(cudaMemsetAsync(b.data(), 0, static_cast<std::size_t>(n) * sizeof(scalar), cudaStreamPerThread),
              "turbulence transport zero");
}
}

void assembleScalarTransport(
    PressureMatrix&             M,
    const DeviceMesh&           dm,
    const DeviceBoundary&       db,
    const DeviceBuffer<scalar>& field,
    const DeviceBuffer<scalar>& gammaFace,
    const DeviceBuffer<scalar>& gammaBnd,
    const TransportScheme&      sc)
{
    const int nC = dm.nCells;

    // fvm::div(phi, field). The boundary half carries the flux-conditional switch the caller has
    // already applied to db.
    //
    // limitedLinear is a WEIGHT change, not a correction, so it replaces the upwind coefficients rather
    // than adding to the source -- the same shape the host closures' divWithScheme takes. The limiter's
    // gradient is the field's own Gauss gradient, limited by the case's grad(<field>) cellLimited
    // coefficient when it names one; the corrected-laplacian block below builds the same three buffers
    // the same way, and this is deliberately the identical call sequence so the two cannot drift.
    if (sc.limitedLinear)
    {
        DeviceBuffer<scalar> bval, gx, gy, gz;
        if (sc.bndValues) deviceCopy(bval, *sc.bndValues);
        else              deviceBCValue(db, field, bval);
        // The limiter's gradient takes the case's OWN gradScheme for this field -- OpenFOAM builds it
        // through fvc::grad(lPhi) (LimitedScheme.C:56-59), not through a scheme the closure chooses.
        if (sc.limGradLeastSq) deviceLeastSquaresGrad(dm, field, bval, gx, gy, gz);
        else                   deviceGaussGrad(dm, field, bval, gx, gy, gz);
        if (sc.limGradK > scalar(0)) deviceCellLimitGrad(dm, field, bval, gx, gy, gz, sc.limGradK);
        deviceDivLimitedCoeffs(dm, *sc.phiInt, field, gx, gy, gz,
                               scalar(2) / std::fmax(sc.limiterCoeff, scalar(1e-15)),
                               M.diag, M.upper, M.lower);
    }
    else
    {
        deviceDivUpwindCoeffs(dm, *sc.phiInt, M.diag, M.upper, M.lower);
    }
    zeroed(M.source, nC);
    deviceBCDivCoeffs(db, *sc.phiBnd, M.iC, M.bC);

    // linearUpwind's explicit correction, part of the same fvm::div object -- SUBTRACTED, because
    // `fvm += fvc::surfaceIntegrate(...)` is `source -= V*su` (fvMatrix.C:1855-1862), exactly as the
    // momentum equation's deviceAxpy(-corrFac, lu, source). Internal faces only: linearUpwind::correction
    // leaves every uncoupled boundary face at zero.
    if (sc.linearUpwind)
    {
        DeviceBuffer<scalar> bval, gx, gy, gz, lu;
        if (sc.bndValues) deviceCopy(bval, *sc.bndValues);
        else              deviceBCValue(db, field, bval);
        deviceGaussGrad(dm, field, bval, gx, gy, gz);
        if (sc.luGradK > scalar(0)) deviceCellLimitGrad(dm, field, bval, gx, gy, gz, sc.luGradK);
        deviceLinearUpwindCorr(dm, *sc.phiInt, gx, gy, gz, lu);
        deviceAxpy(-1.0, lu, M.source);
    }

    // - fvm::laplacian(gamma, field).
    {
        DeviceBuffer<scalar> lDiag, lUp, lLo, lIC, lBC;
        deviceLaplacianCoeffs(dm, gammaFace, lDiag, lUp, lLo, sc.correctedLaplacian);
        deviceBCLaplacianCoeffsFace(db, gammaBnd, lIC, lBC);
        // FP-2: the five `axpy(-1, l, M)` subtractions in one launch (subtractLaplacianKernel above),
        // the same `y += a*x` statement per array, so the same doubles.
        subtractLaplacian(lDiag, lUp, lLo, lIC, lBC, M);

        if (sc.correctedLaplacian)
        {
            DeviceBuffer<scalar> bval, gx, gy, gz, ffc, corr;
            if (sc.bndValues) deviceCopy(bval, *sc.bndValues);
            else              deviceBCValue(db, field, bval);
            // correctedSnGrad's correction takes the field's OWN grad scheme (correctedSnGrad.C:52-55):
            // its base (leastSquares or Gauss linear) and its cellLimited coefficient, both from the
            // case's grad(<field>) entry. The host twin is kEpsilon_cpp.cu's laplacian block.
            if (sc.gradFieldLeastSq) deviceLeastSquaresGrad(dm, field, bval, gx, gy, gz);
            else                     deviceGaussGrad(dm, field, bval, gx, gy, gz);
            if (sc.gradFieldLimitK > scalar(0))
                deviceCellLimitGrad(dm, field, bval, gx, gy, gz, sc.gradFieldLimitK);
            if (sc.snGradLimitCoeff > scalar(0.0))
            {
                deviceLaplacianCorrFluxLimited(dm, gammaFace, field, gx, gy, gz, sc.snGradLimitCoeff, ffc);
                deviceFaceDivSource(dm, ffc, corr);
            }
            else
            {
                deviceLaplacianCorr(dm, gammaFace, gx, gy, gz, corr);
            }
            // deviceLaplacianCorr returns -V*div(faceFluxCorr) -- already negated -- and the laplacian
            // itself enters this equation with -1, so its explicit source does too. The two signs
            // compose to the reference's `L.source -= corr` followed by `M -= L`.
            deviceAxpy(-1.0, corr, M.source);
        }
    }
}

namespace
{
__global__ void wallFacesTakeCellKernel(
    int            nB,
    const label*   wfMask,
    const label*   bndCell,
    const scalar*  cell,
    scalar*        bnd)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nB || !wfMask[f]) return;
    bnd[f] = cell[bndCell[f]];
}
} // namespace

void wallFacesTakeCell(
    const DeviceMesh&           dm,
    const DeviceBuffer<label>&  wfMask,
    const DeviceBuffer<scalar>& field,
    DeviceBuffer<scalar>&       bnd)
{
    const int nB = static_cast<int>(bnd.size());
    if (nB == 0) return;
    const int tpb = 256;
    wallFacesTakeCellKernel<<<(nB + tpb - 1) / tpb, tpb>>>(nB, wfMask.data(), dm.bndCell.data(),
                                                           field.data(), bnd.data());
    cudaCheck(cudaGetLastError(), "turbulence wallFacesTakeCell");
}

void solveScalarEqn(
    PressureMatrix&             M,
    DeviceBuffer<scalar>&       field,
    const DeviceMesh&           dm,
    bool                        relaxEquation,
    scalar                      alpha,
    const DeviceBuffer<label>*  fvoMask,
    const DeviceBuffer<scalar>* fvoVal,
    const DeviceBuffer<label>*  wallMask,
    const DeviceBuffer<scalar>* wallVal,
    const SolveControls&        sv,
    scalar&                     residualOut,
    const std::string&          dumpPrefix,   // "" = no dump; else <dir>/<name> path prefix
    bool                        gs)           // this field's own solver: the case's smoothSolver, or BiCGStab
{
    const int nC  = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nB  = dm.nBndFaces;

    // The guard is "the case NAMES a factor", not "the factor is below 1": fvMatrix::relax early-returns
    // only on alpha <= 0, so relax(1.0) still applies the dominance clamp and adds (D - D0)*psi.
    if (relaxEquation && alpha > scalar(0.0))
    {
        DeviceBuffer<scalar> relaxedDiag, delta, t;
        deviceRelaxDiag(M.view(dm), dm, M.iC, alpha, relaxedDiag, delta);
        deviceCopy(M.diag, relaxedDiag);
        deviceHadamard(t, delta, field);
        deviceAxpy(1.0, t, M.source);
    }

    // Both constraints go through the SAME four kernels, because in OpenFOAM they are the same call --
    // FixedValueConstraint::constrain and epsilonWallFunction::manipulateMatrix both end in
    // fvMatrix::setValues. Order matters and is OpenFOAM's: whichever runs first is the one whose value
    // reaches the neighbours, since setValues zeroes the coefficient it just transferred through.
    auto applySetValues = [&](const DeviceBuffer<label>* mask, const DeviceBuffer<scalar>* val)
    {
        if (!mask || !val) return;
        deviceSetValues(dm, *mask, *val, M.diag, M.upper, M.lower, M.source, M.iC, M.bC, field);
    };
    applySetValues(fvoMask, fvoVal);
    applySetValues(wallMask, wallVal);

    // Fold the boundary coefficients in exactly as fvMatrix::solve does, then solve. BiCGStab, not PCG:
    // upwind convection makes upper != lower, so the matrix is asymmetric and a symmetric solver would
    // be solving a different system.
    DeviceBuffer<scalar> diagC, b, ones;
    deviceFold(dm, M.diag, M.source, M.iC, M.bC, diagC, b);

    DeviceLduView A{};
    A.nCells = dm.nCells;
    A.nInternalFaces = dm.nInternalFaces;
    A.diag = diagC.data();
    A.upper = M.upper.data();
    A.lower = M.lower.data();
    A.owner = dm.owner.data();
    A.nei = dm.nei.data();
    A.ownerStart = dm.ownerStart.data();
    A.losort = dm.losort.data();
    A.losortStart = dm.losortStart.data();

    // Instrument (BRAE_STAGE_DUMP_DIR, see correct()): the FOLDED system as the solver sees it -- diag
    // with the boundary diagonal folded in, source with the boundary source folded in, the two
    // off-diagonals (<prefix>D/Src/Upper/Lower), and the field before and after the solve
    // (<prefix>SolveIn/SolveOut) -- the same convention the host's
    // captureSystem uses (kEpsilon_cpp.cu), so the two arms' assembled systems diff directly.
    auto dump = [&](const char* what, const DeviceBuffer<scalar>& v)
    {
        if (dumpPrefix.empty()) return;
        const std::vector<scalar> h = v.host();
        std::FILE* fp = std::fopen((dumpPrefix + what).c_str(), "w");
        if (!fp) return;
        for (scalar x : h) std::fprintf(fp, "%.17g\n", (double)x);
        std::fclose(fp);
    };
    dump("D", diagC);
    dump("Src", b);
    dump("Upper", M.upper);
    dump("Lower", M.lower);
    dump("SolveIn", field);

    // the ones vector kept across calls (item 63) and the normFactor kept on the device (item 66)
    DeviceBuffer<scalar> dnf;
    deviceNormFactorInto(A, field, b, deviceOnes(nC), dnf);
    // The solver the case asked for (item 58). The view above is internal-face only, which is what the
    // level-scheduled sweep needs; there is no interface to drop silently.
    DeviceSolverPerf perf;
    if (gs && sv.gsColour)
    {
        // FP-1: the case's smoothSolver swept in COLOUR order, one component through the momentum engine.
        // The driver announced the order; refuse rather than run something else without the colouring.
        if (!sv.colouring || !sv.colouring->valid)
            throw std::runtime_error(
                "brae turbulence: the colour-order smoothSolver was selected for a transported scalar but "
                "SolveControls::colouring is null or invalid; refusing rather than running a solver the "
                "notice did not name");
        GSFusedComponent one;
        one.A = &A; one.b = &b; one.psi = &field; one.normFactor = 1.0; one.dNormFactor = dnf.data();
        deviceColourGaussSeidelFused(1, &one, *sv.colouring, sv.tol, sv.relTol, sv.maxIter, sv.minIter,
                                     sv.nSweeps, sv.gsSymmetric, &perf);
    }
    else if (gs)
        deviceSymGaussSeidel(A, b, field, dnf.data(), sv.tol, sv.relTol, sv.maxIter, &perf, sv.minIter,
                             sv.nSweeps, sv.gsSymmetric);
    else
        perf = deviceJacobiBiCGStab(A, b, field, dnf.data(), sv.tol, sv.relTol, sv.maxIter, /*checkEvery=*/1, sv.minIter,
                                    sv.precon, /*amg=*/nullptr, sv.polyDeg);
    residualOut = perf.initialResidual;
    dump("SolveOut", field);
}


} // namespace turbulence
} // namespace gpu
} // namespace brae
