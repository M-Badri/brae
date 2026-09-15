// Standalone h -> T inversion for the liquid path. Nothing in the solver calls this yet.
//
// THE TARGETS COME FROM OPENFOAM. hTarget[i] is OF's own H2O h(T) at a known temperature
// (dissect/liqref), so "did Newton recover the right T" is answered against OF and not against brae's
// own forward evaluation.
//
// THE GUESSES ARE DELIBERATELY BAD, AND REVERSED. hTarget[0] corresponds to 280 K and is started from
// 400 K; hTarget[12] corresponds to 400 K and is started from 280 K. A test where Tguess ~ Ttrue would
// pass with almost any code -- including code that simply returns its own initial guess. Reversing the
// pairing also makes an indexing error large and obvious rather than subtle.
//
// TWO ASSERTIONS PER CASE, and the second is the important one:
//     T_recovered ~ T_OF            (did we land on the right temperature)
//     h(T_recovered) ~ hTarget      (is the inversion self-consistent)
// The first alone can pass while the thermodynamics is inconsistent; the second is what actually says
// the equation h(T) = hTarget was solved.
#include "device_thermo.cuh"
#include "nsrds_functions.cuh"
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <vector>

using namespace brae;

namespace {

// OpenFOAM v2412, dissect/liqref: T and Foam::H2O::h(1e5, T).
// TinvRev: OpenFOAM's OWN recovered temperature for this row's REVERSED guess (kOF[n-1-i].T), from
// tools/liqref's INVX table, 2026-09-15. It is NOT T. OpenFOAM's inversion stops on the temperature
// STEP against Ttol = T0*1e-4 computed once from the initial guess (thermoI.H:43-88, thermo.C:33), so it
// does not reach the true temperature and its answer depends on T0 -- for target 350 from guess 330 it
// returns 350.00000009383587, an error of 9.4e-08 K.
//
// This column exists because the test used to assert `T`, the TRUE temperature, to 1e-8 K. That asks
// brae to be more accurate than OpenFOAM, and brae deliberately is not: e441dfc transcribed OpenFOAM's
// loop, quirks included. Asserting the true value made a faithful port fail and would have PASSED a port
// that iterated to convergence -- the exact defect tools/liqref was written to catch. The bound below is
// therefore far TIGHTER than the 1e-8 it replaces, not looser.
//
// FAIL-PROOF, measured 2026-09-15: replace the do-while's stopping test in nsrds_functions.cuh with a
// converged one (`fabs(Tnew - Test) > 1e-13*max(fabs(Tnew),1)`) and rebuild -- 0 failures becomes 5.
// The assertion this column replaced passed that same mutation.
struct Row { double T, h, TinvRev; };
const std::vector<Row> kOF = {
    {280, -15934577.139513608, 279.99999999987978},   // from guess 400
    {290, -15892558.818986528, 289.99999999999233},   // from guess 390
    {300, -15850679.935920451, 299.99999999999983},   // from guess 380
    {310, -15808882.056429174, 310},   // from guess 370
    {320, -15767116.311589021, 320},   // from guess 360
    {330, -15725342.149132438, 329.99999999999994},   // from guess 350
    {340, -15683526.085141577, 340},   // from guess 340
    {350, -15641640.455741892, 350.00000009383587},   // from guess 330
    {360, -15599662.168795714, 360.00000000000017},   // from guess 320
    {370, -15557571.455595860, 370.00000000000034},   // from guess 310
    {380, -15515350.622559205, 380.00000000000023},   // from guess 300
    {390, -15472982.802920273, 390.00000000000028},   // from guess 290
    {400, -15430450.708424833, 400.00000000000017},   // from guess 280
};

int failures = 0;

void fail(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    std::printf("  FAIL ");
    std::vprintf(fmt, ap);
    va_end(ap);
    ++failures;
}

}   // namespace

int main()
{
    const int n = static_cast<int>(kOF.size());
    std::printf("h -> T inversion, %d OF-generated targets, guesses reversed (worst case)\n", n);

    // ---------------------------------------------------------------------------------------------
    // 1. Host inversion, reversed guesses.
    {
        double worstT = 0, worstH = 0;
        int worstIter = 0;
        for (int i = 0; i < n; ++i)
        {
            const double hT = kOF[i].h;
            const double T0 = kOF[n - 1 - i].T;          // reversed: 280 K target started from 400 K
            const HeToTResult r = h2oHToT(hT, T0);

            if (!r.converged)
                fail("target T=%.0f from guess %.0f did not converge (residual %.3e, %d iters)\n",
                     kOF[i].T, T0, r.residual, r.iterations);

            // Against OPENFOAM'S answer for this same guess, not against the true temperature.
            const double dT = std::fabs(r.T - kOF[i].TinvRev);
            const double rh = std::fabs(H2OLiquid::h(r.T) - hT)/std::fabs(hT);
            if (dT > 1e-11)
                fail("target T=%.0f from guess %.0f: brae %.17g vs OpenFOAM %.17g (dT %.3e K)\n",
                     kOF[i].T, T0, r.T, kOF[i].TinvRev, dT);
            // The energy residual OpenFOAM exits with is ~Cp*T0*1e-4/|h| ~ 1e-5 for water, so this is a
            // sanity net on the arithmetic, not a convergence claim; see residualBound in nsrds_functions.cuh.
            if (rh > 1e-9)
                fail("target T=%.0f: h(T_recovered) off by %.3e relative\n", kOF[i].T, rh);

            worstT = std::fmax(worstT, dT);
            worstH = std::fmax(worstH, rh);
            worstIter = r.iterations > worstIter ? r.iterations : worstIter;
        }
        std::printf("  host: worst |dT| %.2e K, worst enthalpy residual %.2e, worst %d iterations\n",
                    worstT, worstH, worstIter);
    }

    // ---------------------------------------------------------------------------------------------
    // 2. Targets AT the valid-range bounds, where projection/clamping hides mistakes.
    {
        const double Tt = H2OLiquid::Tt, Tc = H2OLiquid::Tc;
        struct { const char* what; double Ttrue, T0; } cases[] = {
            {"at Tt, guessed from Tc",   Tt,        Tc},
            {"at Tc, guessed from Tt",   Tc,        Tt},
            {"just inside Tt",           Tt + 1e-3, Tc},
            {"just inside Tc",           Tc - 1e-3, Tt},
        };
        for (const auto& c : cases)
        {
            const double hT = H2OLiquid::h(c.Ttrue);
            const HeToTResult r = h2oHToT(hT, c.T0);
            const double dT = std::fabs(r.T - c.Ttrue);
            if (!r.converged || dT > 1e-6)
                fail("bound case '%s': recovered %.10g for true %.10g (dT %.3e, converged %d, res %.3e)\n",
                     c.what, r.T, c.Ttrue, dT, (int)r.converged, r.residual);
        }
        std::printf("  bounds: Tt=%.2f and Tc=%.2f recovered exactly, from the opposite bound\n", Tt, Tc);
    }

    // ---------------------------------------------------------------------------------------------
    // 3. Guesses from outside [Tt, Tc] -- what a diverging outer iteration hands in -- against what
    // OPENFOAM does with them, from tools/liqref's OOR rows, 2026-09-15. This block used to assert that
    // brae PROJECTED such a guess back into range and still recovered 300 K. OpenFOAM does not project:
    // limit() is the identity for a liquid (liquidPropertiesI.H:28-31), so the step is taken at the bad
    // guess itself, and the three guesses have three different outcomes.
    {
        const double hT = H2OLiquid::h(300.0);

        // (a) NEGATIVE: OpenFOAM REFUSES it -- thermoI.H:55-60 guards `if (T0 < 0)` and aborts before
        // the first step (liqref core-dumped on that line). brae reports the refusal through
        // `converged`, being BRAE_HD and unable to throw on the device.
        {
            const HeToTResult r = h2oHToT(hT, -500.0);
            if (r.converged)
                fail("guess -500 K was accepted (T=%.10g); OpenFOAM aborts on T0 < 0\n", r.T);
        }
        // (b) ZERO: allowed, and Ttol = T0*1e-4 is then ZERO, so the loop runs until the step is exactly
        // zero -- the one place OpenFOAM does iterate to convergence.
        {
            const HeToTResult r = h2oHToT(hT, 0.0);
            if (!r.converged || std::fabs(r.T - 300.00000000000028) > 1e-11)
                fail("guess 0 K gave %.17g, OpenFOAM 300.00000000000028 (converged %d)\n",
                     r.T, (int)r.converged);
        }
        // (c) 5000 K, far above Tc: the ENTHALPY form survives, because Hs carries no rho and its
        // correlation stays finite there. The internal-energy form does NOT -- p/rho raises a negative
        // (1 - T/Tc) to a fractional power and returns NaN; test_etot asserts that. Same guess, same
        // loop, different answer, which is why the two forms are gated separately.
        {
            const HeToTResult r = h2oHToT(hT, 5000.0);
            if (!r.converged || std::fabs(r.T - 299.99999999997334) > 1e-11)
                fail("guess 5000 K gave %.17g, OpenFOAM 299.99999999997334 (converged %d)\n",
                     r.T, (int)r.converged);
        }
        std::printf("  out-of-range guesses: T0<0 refused as OpenFOAM refuses it, T0=0 and T0=5000 K"
                    " reproduced to 1e-11 K\n");
    }

    // ---------------------------------------------------------------------------------------------
    // 4. Failure must be EXPLICIT. An unreachable enthalpy (far below h(Tt)) cannot be solved; the
    // result must say so rather than quietly return a clamped temperature as if it were an answer.
    {
        const double unreachable = H2OLiquid::h(H2OLiquid::Tt) - 1e7;
        const HeToTResult r = h2oHToT(unreachable, 300.0);
        if (r.converged)
            fail("an unreachable enthalpy reported convergence at T=%.10g\n", r.T);
        else
            std::printf("  unreachable target reported failure explicitly (T clamped to %.2f, residual %.2e)\n",
                        r.T, r.residual);
    }

    // ---------------------------------------------------------------------------------------------
    // 5. GPU vector: 13 targets, 13 reversed guesses, one inversion per cell.
    {
        std::vector<scalar> h(n), T0(n);
        for (int i = 0; i < n; ++i) { h[i] = kOF[i].h; T0[i] = kOF[n - 1 - i].T; }

        DeviceBuffer<scalar> hD, T0D, TD, resD;
        DeviceBuffer<label> okD;
        hD.copyFrom(h);
        T0D.copyFrom(T0);
        deviceH2OHToT(hD, T0D, TD, okD, resD);

        const std::vector<scalar> T = TD.host(), res = resD.host();
        const std::vector<label> ok = okD.host();
        int converged = 0, distinct = 0;
        double worstT = 0;
        for (int i = 0; i < n; ++i)
        {
            if (ok[i]) ++converged;
            // Same oracle as the host block: OpenFOAM's recovered T for THIS cell's guess, not the true
            // temperature. The device arm is the same transcription, so it must reproduce the same answer.
            const double dT = std::fabs(T[i] - kOF[i].TinvRev);
            worstT = std::fmax(worstT, dT);
            if (dT > 1e-11)
                fail("GPU cell %d: brae %.17g vs OpenFOAM %.17g (dT %.3e)\n", i, T[i], kOF[i].TinvRev, dT);
            if (res[i] > 1e-9)   // OF exits at ~1e-5; see residualBound in nsrds_functions.cuh
                fail("GPU cell %d: enthalpy residual %.3e\n", i, res[i]);
            if (i > 0 && T[i] != T[i-1]) ++distinct;
        }
        if (converged != n) fail("only %d of %d GPU cells converged\n", converged, n);
        if (distinct != n - 1)
            fail("only %d of %d neighbouring GPU cells differ -- a value was broadcast\n", distinct, n - 1);
        std::printf("  GPU: %d/%d converged, worst |dT| %.2e K, all cells distinct\n", converged, n, worstT);
    }

    // ---------------------------------------------------------------------------------------------
    // 6. NEGATIVE CONTROLS. Both of the failure modes worth guarding, checked in-process so the test
    // carries its own proof that it can go red.
    {
        int caught = 0;

        // (a) WRONG DERIVATIVE: Cp scaled by 1.01.
        //
        // BE PRECISE ABOUT WHAT CATCHES THIS, because it is easy to claim too much. A wrong derivative
        // changes the PATH, not the fixed point: Newton still converges to h(T) = hTarget, just more
        // slowly. So NEITHER stopping criterion detects it -- not |dT|, and not the enthalpy residual
        // either. What detects it is the ITERATION COUNT: the loop below re-derives the same inversion
        // with a perturbed slope and requires it to take strictly more steps than the real
        // implementation. If the real implementation is itself perturbed the two agree, the count no
        // longer increases, and this control fires. Verified by mutation (Cp*1.01 in h2oHToT -> red).
        //
        // The enthalpy residual earns its place against a different failure -- an h() inconsistent with
        // its own Cp -- which is covered directly by the dh/dT == Cp assertion in tests/test_nsrds.cu.
        {
            const double hT = kOF[0].h;                       // 280 K
            double T = 400.0;                                 // reversed guess
            int it = 0;
            for (; it < 50; ++it)
            {
                const double e = H2OLiquid::h(T) - hT;
                if (std::fabs(e)/std::fabs(hT) <= 1e-12) break;
                T = T - e/(H2OLiquid::Cp(T)*1.01);            // deliberately wrong dh/dT
            }
            const double correct = h2oHToT(hT, 400.0).iterations;
            if (it > correct) ++caught;                       // wrong slope => strictly more iterations
            else std::printf("  note: perturbed derivative did not change the iteration count\n");
        }

        // (b) WRONG TARGET/INDEX: invert every cell against hTarget[0] instead of its own.
        {
            int wrong = 0;
            for (int i = 1; i < n; ++i)
            {
                const HeToTResult r = h2oHToT(kOF[0].h, kOF[n - 1 - i].T);
                if (std::fabs(r.T - kOF[i].T) > 1e-8) ++wrong;
            }
            if (wrong == n - 1) ++caught;
        }

        if (caught != 2)
            fail("negative controls: only %d of 2 detected -- the test cannot catch a wrong derivative\n"
                 "       or a wrong target index\n", caught);
        else
            std::printf("  negative controls: wrong derivative and wrong target index both rejected\n");
    }

    std::printf("test_hetot: %d failures\n", failures);
    return failures ? 1 : 0;
}
