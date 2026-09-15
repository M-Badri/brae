// The FUSED cellLimited gradient is the same limiter, not a similar one.
//
// WHY IT EXISTS (FP-4). nsys on gasMixing/injectorPipe at 74,650 cells (--cuda-graph-trace=node, 20
// iterations, after FP-3): cellLimitGradKernel is 12 launches and 1.90 of the iteration's 13.73 GPU ms,
// and 9 of those launches -- 1.41 ms, 44% of the momentum phase -- are the three velocity components at
// three sites. Each launch re-reads the whole of the addressing and the face offsets and walks every
// face of every cell six times (three loops for the range, three for the limiter) to carry ONE
// component. deviceCellLimitGradFused reads that row once for up to three fields.
//
// WHAT IS AT STAKE. A fused kernel's failure mode is not "wrong formula", it is CROSS-CONTAMINATION:
// one field's range, limiter or gradient leaking into another's. Here that is especially easy to hide,
// because a limiter is a min over faces and a contaminated one is still in [0,1] and still looks like
// a limiter. So the bar is memcmp against the unchanged single-field kernel, which deviceCellLimitGrad
// keeps and does NOT forward to the fused path.
//
// ARMS
//   (a) n = 3 against three deviceCellLimitGrad calls, memcmp on all nine arrays, at k = 1.
//   (b) n = 1 and n = 2 likewise -- the N-dependent unrolling must not change any field's arithmetic.
//   (c) k < 1, which takes the widening branch (maxD/minD stretched before the limiter): same memcmp.
//   (d) CONTROL, THE LIMITER BITES. The limited gradient must DIFFER from the unlimited one on every
//       field, or every arm above is comparing two copies of an unlimited gradient.
//   (e) CONTROL, no cross-contamination. One ulp into field 1's interior, and separately into field 1's
//       boundary values: field 1 moves, fields 0 and 2 do not move by a bit.
//   (f) EMPTY PATCHES. The same mesh with its two z patches typed `empty`: fused and separate agree bit
//       for bit; poisoning the empty patch is inert on BOTH arms (the skip is live in both loops) and
//       the same poison on a real patch moves both.
//   (g) The three fields' limited gradients differ pairwise, so (e) has something to detect.
//
// FAIL-PROOF, RUN: with the fused limiter loop reading field 0's range for every field
// (`limFace(maxD[0], minD[0], ...)`), arms (a), (b) n=2, (c), (e) and (f) went red and the exit code
// was 1; restoring it turned them green again.
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
    std::printf("  %-80s %s\n", what.c_str(), ok ? "ok" : "FAIL");
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
    G3 r; r.x = gx.host(); r.y = gy.host(); r.z = gz.host(); return r;
}

bool sameG3(const G3& a, const G3& b)
{
    return sameBits(a.x, b.x) && sameBits(a.y, b.y) && sameBits(a.z, b.z);
}

// The UNLIMITED Gauss gradient the limiter is applied to: every arm starts from these bits.
void unlimited(const DeviceMesh& dm, const DeviceBuffer<scalar>* vol, const DeviceBuffer<scalar>* bval,
               int n, DeviceBuffer<scalar>* gx, DeviceBuffer<scalar>* gy, DeviceBuffer<scalar>* gz)
{
    for (int i = 0; i < n; ++i) deviceGaussGrad(dm, vol[i], bval[i], gx[i], gy[i], gz[i]);
}

// The reference: n separate deviceCellLimitGrad calls, i.e. the untouched cellLimitGradKernel.
std::vector<G3> separateLimited(const DeviceMesh& dm, int n, const DeviceBuffer<scalar>* vol,
                                const DeviceBuffer<scalar>* bval, scalar k)
{
    DeviceBuffer<scalar> gx[3], gy[3], gz[3];
    unlimited(dm, vol, bval, n, gx, gy, gz);
    std::vector<G3> out;
    for (int i = 0; i < n; ++i)
    {
        deviceCellLimitGrad(dm, vol[i], bval[i], gx[i], gy[i], gz[i], k);
        out.push_back(fetch(gx[i], gy[i], gz[i]));
    }
    return out;
}

// The thing under test: one launch for all n.
std::vector<G3> fusedLimited(const DeviceMesh& dm, int n, const DeviceBuffer<scalar>* vol,
                             const DeviceBuffer<scalar>* bval, scalar k)
{
    DeviceBuffer<scalar> gx[3], gy[3], gz[3];
    unlimited(dm, vol, bval, n, gx, gy, gz);
    const DeviceBuffer<scalar>* vp[3] = {&vol[0], &vol[1], &vol[2]};
    const DeviceBuffer<scalar>* bp[3] = {&bval[0], &bval[1], &bval[2]};
    deviceCellLimitGradFused(dm, n, vp, bp, gx, gy, gz, k);
    std::vector<G3> out;
    for (int i = 0; i < n; ++i) out.push_back(fetch(gx[i], gy[i], gz[i]));
    return out;
}

void runArms(const DeviceMesh& dm,
             const std::vector<std::vector<scalar>>& hVol,
             const std::vector<std::vector<scalar>>& hBnd,
             const std::string& tag)
{
    DeviceBuffer<scalar> vol[3], bval[3];
    for (int i = 0; i < 3; ++i) { vol[i].copyFrom(hVol[i]); bval[i].copyFrom(hBnd[i]); }

    const std::vector<G3> ref = separateLimited(dm, 3, vol, bval, 1.0);

    // (d) the limiter must bite, or every memcmp below compares unlimited gradients
    {
        DeviceBuffer<scalar> gx[3], gy[3], gz[3];
        unlimited(dm, vol, bval, 3, gx, gy, gz);
        bool bites = true;
        for (int i = 0; i < 3; ++i) bites = bites && !sameG3(fetch(gx[i], gy[i], gz[i]), ref[i]);
        check(bites, tag + ": CONTROL the limiter BITES on every field (k=1 moves the gradient)");
    }
    // (g)
    check(!sameG3(ref[0], ref[1]) && !sameG3(ref[1], ref[2]) && !sameG3(ref[0], ref[2]),
          tag + ": the three limited gradients differ pairwise");

    // (a)
    {
        const std::vector<G3> f3 = fusedLimited(dm, 3, vol, bval, 1.0);
        bool ok = true;
        for (int i = 0; i < 3; ++i) ok = ok && sameG3(f3[i], ref[i]);
        check(ok, tag + ": fused n=3 == three deviceCellLimitGrad calls, BIT FOR BIT (9 arrays)");
    }

    // (b)
    {
        bool ok = true;
        for (int i = 0; i < 3; ++i)
        {
            DeviceBuffer<scalar> gx[1], gy[1], gz[1];
            deviceGaussGrad(dm, vol[i], bval[i], gx[0], gy[0], gz[0]);
            const DeviceBuffer<scalar>* vp[1] = {&vol[i]};
            const DeviceBuffer<scalar>* bp[1] = {&bval[i]};
            deviceCellLimitGradFused(dm, 1, vp, bp, gx, gy, gz, 1.0);
            ok = ok && sameG3(fetch(gx[0], gy[0], gz[0]), ref[i]);
        }
        check(ok, tag + ": fused n=1 == deviceCellLimitGrad, bit for bit, for each field");

        const std::vector<G3> f2 = fusedLimited(dm, 2, vol, bval, 1.0);
        check(f2.size() == 2 && sameG3(f2[0], ref[0]) && sameG3(f2[1], ref[1]),
              tag + ": fused n=2 == two deviceCellLimitGrad calls, bit for bit");
    }

    // (c) k < 1: the widening branch
    {
        const scalar k = 0.5;
        const std::vector<G3> refK = separateLimited(dm, 3, vol, bval, k);
        const std::vector<G3> fusK = fusedLimited(dm, 3, vol, bval, k);
        bool ok = true, widened = false;
        for (int i = 0; i < 3; ++i)
        {
            ok = ok && sameG3(fusK[i], refK[i]);
            widened = widened || !sameG3(refK[i], ref[i]);
        }
        check(ok, tag + ": fused n=3 at k=0.5 (the widening branch) == separate, bit for bit");
        check(widened, tag + ": CONTROL k=0.5 really widens (its answer differs from k=1)");
    }

    // (e) CONTROL: one ulp into field 1 only
    {
        std::vector<std::vector<scalar>> pVol = hVol;
        pVol[1][pVol[1].size() / 2] = std::nextafter(pVol[1][pVol[1].size() / 2], 1e30);
        DeviceBuffer<scalar> pv[3];
        for (int i = 0; i < 3; ++i) pv[i].copyFrom(pVol[i]);
        const std::vector<G3> f = fusedLimited(dm, 3, pv, bval, 1.0);
        check(!sameG3(f[1], ref[1]), tag + ": CONTROL one ulp into field 1's interior MOVES field 1");
        check(sameG3(f[0], ref[0]) && sameG3(f[2], ref[2]),
              tag + ": CONTROL ...and leaves fields 0 and 2 bit-identical (no cross-contamination)");
    }
    {
        std::vector<std::vector<scalar>> pBnd = hBnd;
        pBnd[1][0] = std::nextafter(pBnd[1][0], 1e30);   // face 0 is on `inlet`, never an empty patch here
        DeviceBuffer<scalar> pb[3];
        for (int i = 0; i < 3; ++i) pb[i].copyFrom(pBnd[i]);
        const std::vector<G3> f = fusedLimited(dm, 3, vol, pb, 1.0);
        check(!sameG3(f[1], ref[1]), tag + ": CONTROL one ulp into field 1's BOUNDARY values moves field 1");
        check(sameG3(f[0], ref[0]) && sameG3(f[2], ref[2]),
              tag + ": CONTROL ...and leaves fields 0 and 2 bit-identical");
    }
}

} // namespace


int main()
{
    std::printf("== fused cellLimited gradient: N fields in one launch, bit for bit ==\n");

    // Sheared, so the mesh is non-orthogonal: Cf - C is not along a face normal and the extrapolations
    // the limiter ratios are real numbers rather than zeros.
    const PrimitiveMesh m = boxtest::boxMesh(9, 7, 5, 0.3);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const int nC = static_cast<int>(m.nCells());
    const int nB = static_cast<int>(m.nFaces() - m.nInternalFaces());
    std::printf("  box 9x7x5 sheared: %d cells, %d internal faces, %d boundary faces\n",
                nC, static_cast<int>(m.nInternalFaces()), nB);

    // Fields with sharp local extrema, so the limiter actually clamps somewhere: a smooth linear field
    // would leave every limiter at 1 and the arms would be vacuous (arm (d) asserts it is not).
    const std::vector<vector>& C = g.C();
    std::vector<std::vector<scalar>> hVol(3, std::vector<scalar>(nC));
    for (int c = 0; c < nC; ++c)
    {
        hVol[0][c] = 1.0 + C[c].x + 3.0 * std::sin(7.0 * C[c].y) * std::cos(5.0 * C[c].z);
        hVol[1][c] = std::sin(11.0 * C[c].x) * (2.0 + C[c].z) - 4.0 * C[c].y * C[c].y;
        hVol[2][c] = 3.0 - 6.0 * C[c].x * C[c].z + std::cos(9.0 * C[c].y) * (1.0 + C[c].x);
    }
    std::vector<std::vector<scalar>> hBnd(3, std::vector<scalar>(nB));
    for (int f = 0; f < nB; ++f)
    {
        hBnd[0][f] = 0.9 + 0.53 * f - 0.004 * f * f;
        hBnd[1][f] = std::cos(0.37 * f) * 9.0 + 1.5;
        hBnd[2][f] = -2.0 + std::sin(0.61 * f) * 7.5;
    }

    {
        DeviceMesh dm = buildDeviceMesh(m, g, fvp);
        runArms(dm, hVol, hBnd, "plain box");
    }

    // (f) the empty-patch variant
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
              "empty box: the fixture HAS empty faces (else arm (f) is vacuous)");

        DeviceMesh dme = buildDeviceMesh(m, g, ep);
        runArms(dme, hVol, hBnd, "empty box");

        DeviceBuffer<scalar> vol[3], bval[3];
        for (int i = 0; i < 3; ++i) { vol[i].copyFrom(hVol[i]); bval[i].copyFrom(hBnd[i]); }
        const std::vector<G3> ref = separateLimited(dme, 3, vol, bval, 1.0);
        const std::vector<G3> fus = fusedLimited(dme, 3, vol, bval, 1.0);

        std::vector<std::vector<scalar>> poisonEmpty = hBnd;
        for (int i = 0; i < 3; ++i)
            for (int f = firstEmpty; f < firstEmpty + nEmptyFaces / 2; ++f) poisonEmpty[i][f] = 1e30;
        DeviceBuffer<scalar> pe[3];
        for (int i = 0; i < 3; ++i) pe[i].copyFrom(poisonEmpty[i]);
        const std::vector<G3> refPE = separateLimited(dme, 3, vol, pe, 1.0);
        const std::vector<G3> fusPE = fusedLimited(dme, 3, vol, pe, 1.0);
        bool inertRef = true, inertFus = true;
        for (int i = 0; i < 3; ++i)
        {
            inertRef = inertRef && sameG3(refPE[i], ref[i]);
            inertFus = inertFus && sameG3(fusPE[i], fus[i]);
        }
        check(inertRef, "empty box: poisoning the empty patch is inert in deviceCellLimitGrad");
        check(inertFus, "empty box: poisoning the empty patch is inert in the FUSED limiter");

        std::vector<std::vector<scalar>> poisonReal = hBnd;
        for (int i = 0; i < 3; ++i) poisonReal[i][0] = 1e30;   // face 0 is on `inlet`, a real patch
        DeviceBuffer<scalar> pr[3];
        for (int i = 0; i < 3; ++i) pr[i].copyFrom(poisonReal[i]);
        const std::vector<G3> refPR = separateLimited(dme, 3, vol, pr, 1.0);
        const std::vector<G3> fusPR = fusedLimited(dme, 3, vol, pr, 1.0);
        check(!sameG3(refPR[0], ref[0]) && !sameG3(fusPR[0], fus[0]),
              "empty box: CONTROL the same poison on a NON-empty patch moves both arms");
    }

    std::printf(failures ? "FAILED (%d)\n" : "PASSED (%d failures)\n", failures);
    return failures ? 1 : 0;
}
