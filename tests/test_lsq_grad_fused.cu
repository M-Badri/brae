// The FUSED leastSquares gradient is the same fit, not a similar one.
//
// WHY IT EXISTS (FP-3). nsys on gasMixing/injectorPipe at 74,650 cells (--cuda-graph-trace=node, 20
// iterations): the iteration is 2,442 launches and 15.8 GPU ms, and the leastSquares gradient is 10
// lsqInvDdKernel launches at 1.99 ms plus 10 lsqGradKernel at 1.57 -- 3.6 ms, 23% of all GPU work and
// the largest single item after the pressure matrix-vector product. Each of those ten re-reads the
// WHOLE of the addressing, the d vectors and the geometry to carry one field, and each rebuilt the
// inverted dd tensor, which is a property of the MESH (OpenFOAM builds it once as a MeshObject,
// leastSquaresVectors.C). Two levers: cache the tensor on the DeviceMesh (lsqInvDdFor), and fit up to
// three fields in one launch (deviceLeastSquaresGradFused).
//
// WHAT IS AT STAKE. A fused kernel's failure mode is not "wrong formula", it is CROSS-CONTAMINATION:
// one field's cell value, face delta or accumulator leaking into another's, which looks plausible and
// converges. So the bar here is memcmp, not a tolerance -- fusing N independent fields is a loop
// interchange, the same faces in the same order with the same expressions, so the bits must equal N
// separate deviceLeastSquaresGrad launches. The reference is the UNCHANGED lsqGradKernel, and
// deviceLeastSquaresGrad deliberately does NOT forward to the fused path, or these arms would be
// comparing a thing to itself.
//
// ARMS
//   (a) n = 3 against three deviceLeastSquaresGrad calls, memcmp on all nine arrays.
//   (b) n = 1 (each field on its own) and n = 2, likewise -- the N-dependent unrolling must not change
//       any field's arithmetic.
//   (c) THE RAW FORM, which the vector gradient uses to write into the slices of a 9*nC tensor it
//       owns: the same bits as the buffer form, at an offset.
//   (d) CONTROL. One ulp into field 1's interior, and separately into field 1's boundary values: the
//       three field-1 arrays must move and the field-0 and field-2 arrays must not move by a bit.
//   (e) EMPTY PATCHES. The same mesh with its two z patches typed `empty`: the fused and separate arms
//       must still agree bit for bit (the empty faces are in neither dd nor the fit -- that skip is
//       what leaves dd singular in 2-D and safeInv handles), poisoning the empty patch must change
//       NEITHER arm, and the same poison on a non-empty patch must change both.
//   (f) THE CACHED TENSOR. lsqInvDdFor returns the same pointer on a second call, its contents equal
//       a freshly built tensor bit for bit, and a gradient taken after the cache is filled equals one
//       taken before -- the caching must be invisible. CONTROL: a DIFFERENT mesh (the empty-patch
//       variant, whose dd omits those faces) must NOT reuse the first mesh's tensor, which is what a
//       cache keyed on nothing would do.
//   (g) The three fields' gradients differ pairwise, and none is all-zero, or (d) has nothing to see.
//
// FAIL-PROOF, RUN: with the fused boundary loop reading field 0's bval for every field
// (`fld.bval[0][bk]` in place of `fld.bval[i][bk]`), arms (a), (b) n=2, (c), (d-boundary) and (e) went
// red and the exit code was 1; restoring it turned them green again.
#include "box_mesh.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::printf("  %-78s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

bool sameBits(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(scalar)) == 0;
}

struct G3
{
    std::vector<scalar> x, y, z;
};

G3 fetch(const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz)
{
    G3 r;
    r.x = gx.host();
    r.y = gy.host();
    r.z = gz.host();
    return r;
}

bool sameG3(const G3& a, const G3& b)
{
    return sameBits(a.x, b.x) && sameBits(a.y, b.y) && sameBits(a.z, b.z);
}

// The reference: N separate deviceLeastSquaresGrad launches, i.e. the untouched lsqGradKernel.
std::vector<G3> separateGrads(
    const DeviceMesh& dm,
    int n,
    const DeviceBuffer<scalar>* vol,
    const DeviceBuffer<scalar>* bval)
{
    std::vector<G3> out;
    for (int i = 0; i < n; ++i)
    {
        DeviceBuffer<scalar> gx, gy, gz;
        deviceLeastSquaresGrad(dm, vol[i], bval[i], gx, gy, gz);
        out.push_back(fetch(gx, gy, gz));
    }
    return out;
}

// The thing under test: one launch for all n.
std::vector<G3> fusedGrads(
    const DeviceMesh& dm,
    int n,
    const DeviceBuffer<scalar>* vol,
    const DeviceBuffer<scalar>* bval)
{
    const DeviceBuffer<scalar>* vp[3] = {&vol[0], &vol[1], &vol[2]};
    const DeviceBuffer<scalar>* bp[3] = {&bval[0], &bval[1], &bval[2]};
    DeviceBuffer<scalar> gx[3], gy[3], gz[3];
    deviceLeastSquaresGradFused(dm, n, vp, bp, gx, gy, gz);
    std::vector<G3> out;
    for (int i = 0; i < n; ++i)
    {
        out.push_back(fetch(gx[i], gy[i], gz[i]));
    }
    return out;
}

bool allZero(const std::vector<scalar>& v)
{
    for (const scalar s : v)
    {
        if (s != 0.0) return false;
    }
    return true;
}

void runArms(
    const DeviceMesh& dm,
    const std::vector<std::vector<scalar>>& hVol,
    const std::vector<std::vector<scalar>>& hBnd,
    const std::string& tag)
{
    const int nC = dm.nCells;
    DeviceBuffer<scalar> vol[3], bval[3];
    for (int i = 0; i < 3; ++i)
    {
        vol[i].copyFrom(hVol[i]);
        bval[i].copyFrom(hBnd[i]);
    }

    const std::vector<G3> ref = separateGrads(dm, 3, vol, bval);

    // (g)
    check(!allZero(ref[0].x) && !allZero(ref[1].y) && !allZero(ref[2].z),
          tag + ": the reference gradients are not all zero");
    check(!sameG3(ref[0], ref[1]) && !sameG3(ref[1], ref[2]) && !sameG3(ref[0], ref[2]),
          tag + ": the three reference gradients differ pairwise");

    // (a)
    {
        const std::vector<G3> f3 = fusedGrads(dm, 3, vol, bval);
        bool ok = true;
        for (int i = 0; i < 3; ++i)
        {
            ok = ok && sameG3(f3[i], ref[i]);
        }
        check(ok, tag + ": fused n=3 == three deviceLeastSquaresGrad calls, BIT FOR BIT (9 arrays)");
    }

    // (b)
    {
        bool ok = true;
        for (int i = 0; i < 3; ++i)
        {
            const DeviceBuffer<scalar>* vp[1] = {&vol[i]};
            const DeviceBuffer<scalar>* bp[1] = {&bval[i]};
            DeviceBuffer<scalar> gx[1], gy[1], gz[1];
            deviceLeastSquaresGradFused(dm, 1, vp, bp, gx, gy, gz);
            ok = ok && sameG3(fetch(gx[0], gy[0], gz[0]), ref[i]);
        }
        check(ok, tag + ": fused n=1 == deviceLeastSquaresGrad, bit for bit, for each field");

        const std::vector<G3> f2 = fusedGrads(dm, 2, vol, bval);
        check(f2.size() == 2 && sameG3(f2[0], ref[0]) && sameG3(f2[1], ref[1]),
              tag + ": fused n=2 == two deviceLeastSquaresGrad calls, bit for bit");
    }

    // (c) the raw form into the slices of one 9*nC buffer, as deviceLeastSquaresGradU writes grad(U)
    {
        DeviceBuffer<scalar> tensor;
        tensor.resize(static_cast<std::size_t>(9) * nC);
        const scalar* v[3] = {vol[0].data(), vol[1].data(), vol[2].data()};
        const scalar* b[3] = {bval[0].data(), bval[1].data(), bval[2].data()};
        scalar* gx[3];
        scalar* gy[3];
        scalar* gz[3];
        for (int i = 0; i < 3; ++i)
        {
            gx[i] = tensor.data() + (0 * 3 + i) * nC;
            gy[i] = tensor.data() + (1 * 3 + i) * nC;
            gz[i] = tensor.data() + (2 * 3 + i) * nC;
        }
        deviceLeastSquaresGradFusedRaw(dm, 3, v, b, gx, gy, gz);
        const std::vector<scalar> h = tensor.host();
        bool ok = true;
        for (int i = 0; i < 3; ++i)
        {
            ok = ok && std::memcmp(h.data() + (0 * 3 + i) * nC, ref[i].x.data(), nC * sizeof(scalar)) == 0
                    && std::memcmp(h.data() + (1 * 3 + i) * nC, ref[i].y.data(), nC * sizeof(scalar)) == 0
                    && std::memcmp(h.data() + (2 * 3 + i) * nC, ref[i].z.data(), nC * sizeof(scalar)) == 0;
        }
        check(ok, tag + ": the RAW form writes the same bits into a caller-owned 9*nC tensor");
    }

    // (d) CONTROL: one ulp into field 1 only
    {
        std::vector<std::vector<scalar>> pVol = hVol;
        pVol[1][pVol[1].size() / 2] = std::nextafter(pVol[1][pVol[1].size() / 2], 1e30);
        DeviceBuffer<scalar> pv[3];
        for (int i = 0; i < 3; ++i)
        {
            pv[i].copyFrom(pVol[i]);
        }
        const std::vector<G3> f = fusedGrads(dm, 3, pv, bval);
        check(!sameG3(f[1], ref[1]),
              tag + ": CONTROL one ulp into field 1's interior MOVES field 1's gradient");
        check(sameG3(f[0], ref[0]) && sameG3(f[2], ref[2]),
              tag + ": CONTROL ...and leaves fields 0 and 2 bit-identical (no cross-contamination)");
    }
    {
        std::vector<std::vector<scalar>> pBnd = hBnd;
        pBnd[1][0] = std::nextafter(pBnd[1][0], 1e30);   // face 0 is on `inlet`, never an empty patch here
        DeviceBuffer<scalar> pb[3];
        for (int i = 0; i < 3; ++i)
        {
            pb[i].copyFrom(pBnd[i]);
        }
        const std::vector<G3> f = fusedGrads(dm, 3, vol, pb);
        check(!sameG3(f[1], ref[1]),
              tag + ": CONTROL one ulp into field 1's BOUNDARY values moves field 1's gradient");
        check(sameG3(f[0], ref[0]) && sameG3(f[2], ref[2]),
              tag + ": CONTROL ...and leaves fields 0 and 2 bit-identical");
    }
}

std::vector<scalar> readTensor(const scalar* p, int nC)
{
    std::vector<scalar> h(static_cast<std::size_t>(6) * nC);
    cudaMemcpy(h.data(), p, h.size() * sizeof(scalar), cudaMemcpyDeviceToHost);
    return h;
}

} // namespace


int main()
{
    std::printf("== fused leastSquares gradient: N fields in one launch, bit for bit ==\n");

    // Sheared, so the mesh is non-orthogonal, the face weights are not all 0.5 and the dd tensor is
    // not diagonal -- a fit that dropped the off-diagonal terms would still pass on a cube.
    const PrimitiveMesh m = boxtest::boxMesh(9, 7, 5, 0.3);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const int nC = static_cast<int>(m.nCells());
    const int nB = static_cast<int>(m.nFaces() - m.nInternalFaces());
    std::printf("  box 9x7x5 sheared: %d cells, %d internal faces, %d boundary faces\n",
                nC, static_cast<int>(m.nInternalFaces()), nB);

    const std::vector<vector>& C = g.C();
    std::vector<std::vector<scalar>> hVol(3, std::vector<scalar>(nC));
    for (int c = 0; c < nC; ++c)
    {
        hVol[0][c] = 1.0 + C[c].x + 0.5 * C[c].y * C[c].y - 0.25 * C[c].z;
        hVol[1][c] = std::sin(0.31 * C[c].x) * (2.0 + C[c].z) - 0.7 * C[c].y;
        hVol[2][c] = 3.0 - 0.4 * C[c].x * C[c].z + std::cos(0.23 * C[c].y);
    }
    std::vector<std::vector<scalar>> hBnd(3, std::vector<scalar>(nB));
    for (int f = 0; f < nB; ++f)
    {
        hBnd[0][f] = 0.9 + 0.13 * f - 0.001 * f * f;
        hBnd[1][f] = std::cos(0.17 * f) * 4.0 + 1.5;
        hBnd[2][f] = -2.0 + std::sin(0.41 * f) * 0.75;
    }

    DeviceMesh dmPlain = buildDeviceMesh(m, g, fvp);
    runArms(dmPlain, hVol, hBnd, "plain box");

    // (f) the cached inverted dd tensor
    {
        DeviceBuffer<scalar> vol0, bval0;
        vol0.copyFrom(hVol[0]);
        bval0.copyFrom(hBnd[0]);

        DeviceMesh dmA = buildDeviceMesh(m, g, fvp);
        const scalar* p1 = lsqInvDdFor(dmA);
        const std::vector<scalar> t1 = readTensor(p1, nC);
        const scalar* p2 = lsqInvDdFor(dmA);
        check(p1 == p2, "cache: a second request returns the SAME buffer, not a rebuilt one");
        check(sameBits(readTensor(p2, nC), t1), "cache: ...holding the same bits");
        check(!allZero(t1), "cache: the tensor is not all zero (else the arms above are vacuous)");

        // a gradient taken before the cache existed and one taken after must agree
        DeviceMesh dmB = buildDeviceMesh(m, g, fvp);
        DeviceBuffer<scalar> ax, ay, az, bx, by, bz;
        deviceLeastSquaresGrad(dmB, vol0, bval0, ax, ay, az);     // fills dmB's cache
        deviceLeastSquaresGrad(dmB, vol0, bval0, bx, by, bz);     // reads it
        check(sameG3(fetch(ax, ay, az), fetch(bx, by, bz)),
              "cache: the second gradient on a mesh equals the first, bit for bit");

        // CONTROL: a mesh whose dd DIFFERS must get its own tensor. The empty-patch variant omits
        // those faces from dd, so a cache that handed back the plain box's would be caught here.
        std::vector<FvPatch> ep = fvp;
        for (std::size_t pi = 0; pi < ep.size(); ++pi)
        {
            if (ep[pi].name == "wallZmin" || ep[pi].name == "wallZmax") ep[pi].type = "empty";
        }
        DeviceMesh dmE = buildDeviceMesh(m, g, ep);
        const std::vector<scalar> tE = readTensor(lsqInvDdFor(dmE), nC);
        check(!sameBits(tE, t1),
              "cache CONTROL: the empty-patch mesh gets its OWN tensor, not the plain box's");
    }

    // (e) the empty-patch variant
    {
        std::vector<FvPatch> ep = fvp;
        int nEmptyFaces = 0, firstEmpty = -1;
        for (std::size_t pi = 0; pi < ep.size(); ++pi)
        {
            if (ep[pi].name == "wallZmin" || ep[pi].name == "wallZmax")
            {
                ep[pi].type = "empty";
                if (firstEmpty < 0) firstEmpty = static_cast<int>(ep[pi].start - m.nInternalFaces());
                nEmptyFaces += static_cast<int>(ep[pi].size);
            }
        }
        std::printf("  empty-patch variant: %d of %d boundary faces are on an `empty` patch\n",
                    nEmptyFaces, nB);
        check(nEmptyFaces > 0 && firstEmpty >= 0,
              "empty box: the fixture HAS empty faces (else arm (e) is vacuous)");

        DeviceMesh dme = buildDeviceMesh(m, g, ep);
        runArms(dme, hVol, hBnd, "empty box");

        DeviceBuffer<scalar> vol[3], bval[3];
        for (int i = 0; i < 3; ++i)
        {
            vol[i].copyFrom(hVol[i]);
            bval[i].copyFrom(hBnd[i]);
        }
        const std::vector<G3> ref = separateGrads(dme, 3, vol, bval);
        const std::vector<G3> fus = fusedGrads(dme, 3, vol, bval);

        std::vector<std::vector<scalar>> poisonEmpty = hBnd;
        for (int i = 0; i < 3; ++i)
        {
            for (int f = firstEmpty; f < firstEmpty + nEmptyFaces / 2; ++f)
            {
                poisonEmpty[i][f] = 1e30;
            }
        }
        DeviceBuffer<scalar> pe[3];
        for (int i = 0; i < 3; ++i)
        {
            pe[i].copyFrom(poisonEmpty[i]);
        }
        const std::vector<G3> refPE = separateGrads(dme, 3, vol, pe);
        const std::vector<G3> fusPE = fusedGrads(dme, 3, vol, pe);
        bool inertRef = true, inertFus = true;
        for (int i = 0; i < 3; ++i)
        {
            inertRef = inertRef && sameG3(refPE[i], ref[i]);
            inertFus = inertFus && sameG3(fusPE[i], fus[i]);
        }
        check(inertRef, "empty box: poisoning the empty patch is inert in deviceLeastSquaresGrad");
        check(inertFus, "empty box: poisoning the empty patch is inert in the FUSED fit");

        std::vector<std::vector<scalar>> poisonReal = hBnd;
        for (int i = 0; i < 3; ++i)
        {
            poisonReal[i][0] = 1e30;   // face 0 is on `inlet`, a real patch
        }
        DeviceBuffer<scalar> pr[3];
        for (int i = 0; i < 3; ++i)
        {
            pr[i].copyFrom(poisonReal[i]);
        }
        const std::vector<G3> refPR = separateGrads(dme, 3, vol, pr);
        const std::vector<G3> fusPR = fusedGrads(dme, 3, vol, pr);
        check(!sameG3(refPR[0], ref[0]) && !sameG3(fusPR[0], fus[0]),
              "empty box: CONTROL the same poison on a NON-empty patch moves both arms");
    }

    std::printf(failures ? "FAILED (%d)\n" : "PASSED (%d failures)\n", failures);
    return failures ? 1 : 0;
}
