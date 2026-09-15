// FACTORY EXPERIMENT: produce the forward and backward walk-order operands together.  The two output
// elements are independent and each expression is identical to the production permutation kernel;
// this removes one launch per coefficient refresh and one launch per component's cell refresh.
__global__ void gsPermuteCoefBothK(
    int nEnt,
    const label* __restrict__ fwdSrc, const label* __restrict__ bwdSrc,
    const scalar* __restrict__ lower, const scalar* __restrict__ upper, int nF,
    scalar* __restrict__ fwdCoef, scalar* __restrict__ bwdCoef)
{
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= nEnt) return;
    const label fs = fwdSrc[e];
    const label bs = bwdSrc[e];
    fwdCoef[e] = (fs < nF) ? lower[fs] : upper[fs - nF];
    bwdCoef[e] = (bs < nF) ? lower[bs] : upper[bs - nF];
}

__global__ void gsPermuteCellBothK(
    int nCells,
    const label* __restrict__ fwdCells, const label* __restrict__ bwdCells,
    const scalar* __restrict__ diag, const scalar* __restrict__ b,
    scalar* __restrict__ fwdDiag, scalar* __restrict__ bwdDiag,
    scalar* __restrict__ fwdB, scalar* __restrict__ bwdB)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    const label fc = fwdCells[i];
    const label bc = bwdCells[i];
    fwdDiag[i] = diag[fc];
    fwdB[i] = b[fc];
    bwdDiag[i] = diag[bc];
    bwdB[i] = b[bc];
}
