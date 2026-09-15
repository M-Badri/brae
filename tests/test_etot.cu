// Standalone e -> T inversion (sensibleInternalEnergy), the form squareBendLiq actually transports.
//
// WHY THIS IS A SEPARATE TEST FROM test_hetot.cu. Same Newton loop, different F and dF/dT. The enthalpy
// test pins the h specialization; this one pins the internal-energy specialization and, critically, its
// PRESSURE dependence -- Es = Hs - p/rho, so an implementation that ignored p would still pass every
// enthalpy assertion.
//
// THE ORACLE IS OPENFOAM, through the exact type squareBendLiq's thermo is built from:
//   thermophysicalPropertiesSelector<liquidProperties>  (basic/rhoThermo/liquidThermo.H)
// so Es and Cv below are computed BY OF, not composed here from parts. 13 temperatures x 4 pressures,
// because a single pressure cannot distinguish e(T) from e(p,T).
//
// TWO FACTS READ FROM OF SOURCE, NOT INFERRED -- both counterintuitive enough to be worth naming:
//   liquidProperties::CpMCv(p,T) { return 0; }   so  Cv == Cp  for a liquid (liquidPropertiesI.H:104)
//   selector::Es(p,T) = Hs(p,T) - p/rho(p,T)                            (…SelectorI.H:152)
// OF then uses Cv as the Newton derivative even though d(Es)/dT also carries p*rho'/rho^2. We reproduce
// that choice; the fixed point is unaffected, and accepting on the residual is what makes that safe.
//
// FAIL-PROOF, measured 2026-09-15: replace the do-while's stopping test with a converged one
// (`fabs(Tnew - Test) > 1e-13*max(fabs(Tnew),1)`) in nsrds_functions.cuh and rebuild -- 0 failures
// becomes 78. That is the mutation the previous assertion, true T to 1e-8 K, PASSED: a hard-converging
// inversion returns exactly 400 where OpenFOAM returns 400.00000000022771.
#include "device_thermo.cuh"
#include "nsrds_functions.cuh"
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <vector>

using namespace brae;

namespace {

// TinvRev: OpenFOAM's OWN recovered temperature when this row's energy is inverted from the REVERSED
// guess used below (680 - T, the far end of the same pressure block), from tools/liqref's INVX table,
// 2026-09-15. It is NOT T, and the gap is larger than the enthalpy form's: OpenFOAM stops on the
// temperature STEP against Ttol = T0*1e-4 fixed from the initial guess (thermoI.H:43-88, thermo.C:33),
// so at p = 5e5 the 350 K row started from 330 K comes back as 350.00000113689094, 1.1e-06 K out.
// The old assertion -- true T to 1e-8 K -- therefore asked brae to be a hundred times more accurate
// than OpenFOAM, and a port that iterated to convergence would have PASSED it while diverging from
// OpenFOAM's answer. The bound below is 1e-11, three orders TIGHTER than the 1e-8 it replaces.
struct Row { double T, p, Es, Cv, TinvRev; };
const std::vector<Row> kOF = {
    {280, 50000, -15934627.158276683, 4211.0158121564882, 279.99999999782392},   // from guess 400
    {290, 50000, -15892608.964519063, 4193.8003320010603, 289.99999999935358},   // from guess 390
    {300, 50000, -15850730.211861541, 4182.9480988064988, 299.99999999985869},   // from guess 380
    {310, 50000, -15808932.466631029, 4177.4402009991136, 309.99999999998698},   // from guess 370
    {320, 50000, -15767166.860136222, 4176.3825576463487, 320.0000000000029},   // from guess 360
    {330, 50000, -15725392.840363575, 4179.0059184567817, 330.00000000000114},   // from guess 350
    {340, 50000, -15683576.923673673, 4184.6658637801202, 340},   // from guess 340
    {350, 50000, -15641691.44649804, 4192.8428046072058, 350.00000020051755},   // from guess 330
    {360, 50000, -15599713.317036513, 4203.1419825700141, 360.00000000000614},   // from guess 320
    {370, 50000, -15557622.766955297, 4215.2934699416455, 370.0000000000183},   // from guess 310
    {380, 50000, -15515402.103085814, 4229.1521696363416, 380.00000000002228},   // from guess 300
    {390, 50000, -15473034.459124567, 4244.6978152094707, 390.00000000000307},   // from guess 290
    {400, 50000, -15430502.547334258, 4262.0349708575422, 400.00000000007094},   // from guess 280
    {280, 100000, -15934677.177039757, 4211.0158121564882, 279.99999999579609},   // from guess 400
    {290, 100000, -15892659.110051598, 4193.8003320010603, 289.99999999873557},   // from guess 390
    {300, 100000, -15850780.487802632, 4182.9480988064988, 299.99999999973136},   // from guess 380
    {310, 100000, -15808982.876832884, 4177.4402009991136, 309.99999999998215},   // from guess 370
    {320, 100000, -15767217.408683423, 4176.3825576463487, 320.00000000000904},   // from guess 360
    {330, 100000, -15725443.531594714, 4179.0059184567817, 330.00000000000199},   // from guess 350
    {340, 100000, -15683627.762205768, 4184.6658637801202, 340},   // from guess 340
    {350, 100000, -15641742.437254189, 4192.8428046072058, 350.00000030667155},   // from guess 330
    {360, 100000, -15599764.465277312, 4203.1419825700141, 360.00000000001495},   // from guess 320
    {370, 100000, -15557674.078314736, 4215.2934699416455, 370.00000000004053},   // from guess 310
    {380, 100000, -15515453.583612423, 4229.1521696363416, 380.00000000005002},   // from guess 300
    {390, 100000, -15473086.115328861, 4244.6978152094707, 390.00000000000682},   // from guess 290
    {400, 100000, -15430554.386243684, 4262.0349708575422, 400.00000000013051},   // from guess 280
    {280, 200000, -15934777.214565905, 4211.0158121564882, 279.99999999182694},   // from guess 400
    {290, 200000, -15892759.401116669, 4193.8003320010603, 289.99999999756244},   // from guess 390
    {300, 200000, -15850881.039684812, 4182.9480988064988, 299.99999999952007},   // from guess 380
    {310, 200000, -15809083.697236596, 4177.4402009991136, 309.99999999999636},   // from guess 370
    {320, 200000, -15767318.505777823, 4176.3825576463487, 320.00000000003308},   // from guess 360
    {330, 200000, -15725544.91405699, 4179.0059184567817, 330.00000000000813},   // from guess 350
    {340, 200000, -15683729.43926996, 4184.6658637801202, 340},   // from guess 340
    {350, 200000, -15641844.418766484, 4192.8428046072058, 350.00000051739556},   // from guess 330
    {360, 200000, -15599866.76175891, 4203.1419825700141, 360.00000000004053},   // from guess 320
    {370, 200000, -15557776.701033611, 4215.2934699416455, 370.00000000010044},   // from guess 310
    {380, 200000, -15515556.544665644, 4229.1521696363416, 380.00000000012074},   // from guess 300
    {390, 200000, -15473189.427737448, 4244.6978152094707, 390.00000000002018},   // from guess 290
    {400, 200000, -15430658.064062536, 4262.0349708575422, 400.00000000021885},   // from guess 280
    {280, 500000, -15935077.327144351, 4211.0158121564882, 279.99999998060042},   // from guess 400
    {290, 500000, -15893060.274311883, 4193.8003320010603, 289.99999999454707},   // from guess 390
    {300, 500000, -15851182.695331354, 4182.9480988064988, 299.99999999922113},   // from guess 380
    {310, 500000, -15809386.158447728, 4177.4402009991136, 310.00000000023419},   // from guess 370
    {320, 500000, -15767621.797061026, 4176.3825576463487, 320.00000000019241},   // from guess 360
    {330, 500000, -15725849.061443819, 4179.0059184567817, 330.00000000004746},   // from guess 350
    {340, 500000, -15684034.470462536, 4184.6658637801202, 340},   // from guess 340
    {350, 500000, -15642150.363303373, 4192.8428046072058, 350.00000113689094},   // from guess 330
    {360, 500000, -15600173.651203705, 4203.1419825700141, 360.00000000018173},   // from guess 320
    {370, 500000, -15558084.569190238, 4215.2934699416455, 370.00000000039347},   // from guess 310
    {380, 500000, -15515865.427825302, 4229.1521696363416, 380.00000000045861},   // from guess 300
    {390, 500000, -15473499.364963213, 4244.6978152094707, 390.00000000009248},   // from guess 290
    {400, 500000, -15430969.09751909, 4262.0349708575422, 400.00000000022771},   // from guess 280
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
    std::printf("e -> T inversion (sensibleInternalEnergy), %d OF (T,p) points across 4 pressures\n", n);
    const EnergyForm E = EnergyForm::sensibleInternalEnergy;

    // 1. Forward: brae's Es(p,T) and Cv must equal OF's, before any inversion is trusted.
    {
        double worstE = 0, worstC = 0;
        for (const Row& r : kOF)
        {
            const double e  = h2oEnergy(E, r.p, r.T);
            const double cv = h2oCpv(E, r.p, r.T);
            const double re = std::fabs(e - r.Es)/std::fabs(r.Es);
            const double rc = std::fabs(cv - r.Cv)/std::fabs(r.Cv);
            if (re > 1e-13) fail("Es(p=%.0f,T=%.0f) brae %.17g OF %.17g rel %.3e\n", r.p, r.T, e, r.Es, re);
            if (rc > 1e-13) fail("Cv(p=%.0f,T=%.0f) brae %.17g OF %.17g rel %.3e\n", r.p, r.T, cv, r.Cv, rc);
            worstE = std::fmax(worstE, re);
            worstC = std::fmax(worstC, rc);
        }
        std::printf("  forward: Es worst %.2e, Cv worst %.2e vs OF\n", worstE, worstC);
    }

    // 2. Cv == Cp for a liquid, asserted directly against OF's own Cv column.
    {
        for (const Row& r : kOF)
            if (std::fabs(H2OLiquid::Cp(r.T) - r.Cv)/r.Cv > 1e-13)
            { fail("Cp != OF's Cv at T=%.0f -- CpMCv is not 0 as OF states\n", r.T); break; }
        std::printf("  Cv == Cp confirmed against OF at every (p,T) (CpMCv = 0)\n");
    }

    // 3. Inversion with REVERSED guesses, per pressure.
    {
        double worstT = 0, worstR = 0;
        int worstIt = 0;
        for (int i = 0; i < n; ++i)
        {
            const Row& r = kOF[i];
            // reversed within this pressure block: 280 K target started from 400 K and vice versa
            const int blk = i/13, j = i%13;
            const double T0 = kOF[blk*13 + (12 - j)].T;
            const HeToTResult res = h2oEnergyToT(E, r.Es, r.p, T0);
            if (!res.converged)
                fail("p=%.0f T=%.0f from %.0f did not converge (res %.3e)\n", r.p, r.T, T0, res.residual);
            // Against OPENFOAM'S answer for this same guess, not against the true temperature.
            const double dT = std::fabs(res.T - r.TinvRev);
            if (dT > 1e-11)
                fail("p=%.0f T=%.0f from %.0f: brae %.17g vs OpenFOAM %.17g (dT %.3e)\n",
                     r.p, r.T, T0, res.T, r.TinvRev, dT);
            // OpenFOAM exits at an energy residual of ~Cv*T0*1e-4/|Es| ~ 1e-5, so this is a sanity net
            // on the arithmetic, not a convergence claim; see residualBound in nsrds_functions.cuh.
            const double rr = std::fabs(h2oEnergy(E, r.p, res.T) - r.Es)/std::fabs(r.Es);
            if (rr > 1e-9) fail("p=%.0f T=%.0f: e(T_recovered) off by %.3e\n", r.p, r.T, rr);
            worstT = std::fmax(worstT, dT);
            worstR = std::fmax(worstR, rr);
            worstIt = res.iterations > worstIt ? res.iterations : worstIt;
        }
        std::printf("  inversion: worst |dT| %.2e K, worst residual %.2e, worst %d iterations\n",
                    worstT, worstR, worstIt);
    }

    // 4. Guesses from outside [Tt, Tc], against what OPENFOAM does with them -- tools/liqref's OOR rows,
    // 2026-09-15. This used to assert that brae PROJECTED a bad guess back into range and still recovered
    // 300 K. OpenFOAM does not project: limit() is the identity for a liquid (liquidPropertiesI.H:28-31),
    // so the first Newton step is taken at the bad guess itself, and the three guesses have three
    // different outcomes rather than one.
    {
        const double p = 1e5, eT = h2oEnergy(E, p, 300.0);

        // (a) NEGATIVE: OpenFOAM REFUSES it. thermoI.H:55-60 guards `if (T0 < 0)` and aborts before the
        // first step; liqref core-dumped on that line. brae reports the refusal through `converged`
        // because it is BRAE_HD and cannot throw on the device.
        {
            const HeToTResult r = h2oEnergyToT(E, eT, p, -500.0);
            if (r.converged)
                fail("guess -500 K was accepted (T=%.10g); OpenFOAM aborts on T0 < 0\n", r.T);
        }
        // (b) ZERO: allowed, and Ttol = T0*1e-4 is then ZERO, so the loop runs until the step is exactly
        // zero -- OpenFOAM converges harder here than anywhere else in this file.
        {
            const HeToTResult r = h2oEnergyToT(E, eT, p, 0.0);
            if (!r.converged || std::fabs(r.T - 300.00000000000034) > 1e-11)
                fail("guess 0 K gave %.17g, OpenFOAM 300.00000000000034 (converged %d)\n",
                     r.T, (int)r.converged);
        }
        // (c) 5000 K, ABOVE Tc: the internal-energy form goes NaN, because rho's correlation raises a
        // negative (1 - T/Tc) to a fractional power at the very first evaluation. OpenFOAM returns that
        // NaN into the temperature field; brae returns the same NaN and declines to call it converged.
        // The ENTHALPY form survives the same guess (test_hetot asserts 299.99999999997334) -- Hs has no
        // rho in it. That asymmetry is the whole reason this file exists separately.
        {
            const HeToTResult r = h2oEnergyToT(E, eT, p, 5000.0);
            if (r.converged || !std::isnan(r.T))
                fail("guess 5000 K gave %.17g converged %d; OpenFOAM returns NaN here\n",
                     r.T, (int)r.converged);
        }
        const HeToTResult bad = h2oEnergyToT(E, h2oEnergy(E, p, H2OLiquid::Tt) - 1e7, p, 300.0);
        if (bad.converged) fail("unreachable internal energy reported convergence at %.10g\n", bad.T);
        std::printf("  T0<0 refused as OpenFOAM refuses it, T0=0 and T0=5000 K reproduced;"
                    " unreachable target failed explicitly\n");
    }

    // 5. GPU vector, with a per-cell pressure field.
    {
        std::vector<scalar> e(n), pf(n), T0(n);
        for (int i = 0; i < n; ++i)
        {
            const int blk = i/13, j = i%13;
            e[i]  = kOF[i].Es;
            pf[i] = kOF[i].p;
            T0[i] = kOF[blk*13 + (12 - j)].T;
        }
        DeviceBuffer<scalar> eD, pD, T0D, TD, resD;
        DeviceBuffer<label> okD;
        eD.copyFrom(e); pD.copyFrom(pf); T0D.copyFrom(T0);
        deviceH2OEnergyToT(E, eD, &pD, T0D, TD, okD, resD);
        const std::vector<scalar> T = TD.host();
        const std::vector<label> ok = okD.host();
        int conv = 0;
        double worstT = 0;
        for (int i = 0; i < n; ++i)
        {
            if (ok[i]) ++conv;
            // Same oracle as the host block: OpenFOAM's recovered T for THIS cell's guess.
            const double dT = std::fabs(T[i] - kOF[i].TinvRev);
            worstT = std::fmax(worstT, dT);
            if (dT > 1e-11)
                fail("GPU cell %d (p=%.0f): brae %.17g vs OpenFOAM %.17g (dT %.3e)\n",
                     i, kOF[i].p, T[i], kOF[i].TinvRev, dT);
        }
        if (conv != n) fail("only %d of %d GPU cells converged\n", conv, n);
        std::printf("  GPU: %d/%d converged across 4 pressures, worst |dT| %.2e K\n", conv, n, worstT);
    }

    // 6. NEGATIVE CONTROLS. The pressure one is the point of this file: it is the only mutation that
    // distinguishes the internal-energy inversion from the enthalpy one.
    {
        int caught = 0;
        // (a) WRONG PRESSURE: invert a 5e5 Pa target at 1e5 Pa.
        {
            int wrong = 0;
            for (int i = 39; i < 52; ++i)      // the 5e5 Pa block
            {
                const HeToTResult r = h2oEnergyToT(E, kOF[i].Es, 1e5, 300.0);
                if (std::fabs(r.T - kOF[i].T) > 1e-8) ++wrong;
            }
            if (wrong == 13) ++caught;
            else std::printf("  note: wrong pressure changed only %d of 13 recovered temperatures\n", wrong);
        }
        // (b) ENTHALPY FORM used for an internal-energy target.
        {
            int wrong = 0;
            for (int i = 39; i < 52; ++i)
            {
                const HeToTResult r = h2oEnergyToT(EnergyForm::sensibleEnthalpy, kOF[i].Es, kOF[i].p, 300.0);
                if (std::fabs(r.T - kOF[i].T) > 1e-8) ++wrong;
            }
            if (wrong == 13) ++caught;
        }
        // (c) WRONG TARGET INDEX.
        {
            int wrong = 0;
            for (int i = 1; i < 13; ++i)
            {
                const HeToTResult r = h2oEnergyToT(E, kOF[0].Es, kOF[0].p, 300.0);
                if (std::fabs(r.T - kOF[i].T) > 1e-8) ++wrong;
            }
            if (wrong == 12) ++caught;
        }
        if (caught != 3)
            fail("negative controls: only %d of 3 detected (pressure / energy form / target index)\n", caught);
        else
            std::printf("  negative controls: wrong pressure, wrong energy form, wrong target all rejected\n");
    }

    std::printf("test_etot: %d failures\n", failures);
    return failures ? 1 : 0;
}
