// INTEGRATION: the real deviceThermoCorrect() liquid path, not the standalone functions.
//
// WHAT THIS CATCHES THAT THE STANDALONE TESTS CANNOT. test_etot proves the inversion solves
// e(p,T) = target; test_liquid_fields proves the correlations fill fields correctly. Neither says the
// two are WIRED together right. This calls the function the solver calls and checks the whole result,
// so it fails on:
//   - passing enthalpy semantics where the case asked for internal energy
//   - handing the inversion the wrong p
//   - evaluating the properties BEFORE updating T (a one-iteration lag that looks perfectly smooth)
//   - writing the solver rho instead of rhoThermo
//   - indexing the wrong field
//
// The inputs are OF's own Es(p,T) at known temperatures, with deliberately WRONG initial temperatures
// in th.T, so the inversion has to do real work and cannot pass by echoing its guess.
//
// FAIL-PROOF, measured 2026-09-15: swap the inversion's stopping test in nsrds_functions.cuh for a
// converged one (`> 1e-13*max(fabs(Tnew),1)`) and rebuild -- 0 failures becomes 147, worst relative
// 2.36e-16 becomes 1.55e-08. The old right-hand side, properties at the TRUE temperature, passed that
// mutation and failed the faithful port.
#include "device_thermo.cuh"
#include "nsrds_functions.cuh"
#include "thermo_types.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace brae;

namespace {
// EVERY NUMBER ON THE RIGHT IS EVALUATED AT Trec, NOT AT T. heRhoThermo::calculate() (heRhoThermo.C:
// 79-91) inverts he -> T and then evaluates rho, mu and alphah at the temperature THAT INVERSION
// RETURNED. OpenFOAM's inversion stops on the temperature step against Ttol = T0*1e-4 fixed from the
// initial guess (thermoI.H:43-88), so Trec is not T -- from the reversed guess this fixture uses it is
// out by up to 1.1e-06 K at p = 5e5 -- and OpenFOAM's own properties inherit that.
//
// Comparing brae at Trec against OpenFOAM's properties at T, which this file used to do, charges brae
// for OpenFOAM's path: mu is exp(a + b/T + ...) with b = 3670.6, so |dln(mu)/dT| ~ 0.047 /K and
// 1.1e-06 K becomes 5e-08 relative, fifty times the tolerance below. Worse, it would have PASSED a port
// that iterated to convergence and FAILED the faithful one. T, Trec and the properties at Trec all come
// from tools/liqref's CORR block, which is OpenFOAM composing calculate() on itself. 2026-09-15.
struct Row { double T, p, Es, Trec, rho, mu, kappa, Cp, alphah; };
const std::vector<Row> kOF = {
    {280, 50000, -15934627.158276683,
     279.99999999782392, 999.62487928508438, 0.0014300291330041545, 0.57871727999644518, 4211.0158121610057, 0.00013742937709356628},
    {290, 50000, -15892608.964519063,
     289.99999999935358, 997.09779658930358, 0.0011126737698883059, 0.59440638499902732, 4193.8003320019579, 0.00014173454574440379},
    {300, 50000, -15850730.211861541,
     299.99999999985869, 994.5114684213421, 0.00088614741034161735, 0.60880999999980545, 4182.9480988066089, 0.00014554567391680007},
    {310, 50000, -15808932.466631029,
     309.99999999998698, 991.86272142792006, 0.00072058248924846487, 0.62193901499998361, 4177.4402009991181, 0.00014888041122676862},
    {320, 50000, -15767166.860136222,
     320.0000000000029, 989.14811145249303, 0.00059698641565938872, 0.63380432000000309, 4176.3825576463478, 0.00015175916268484548},
    {330, 50000, -15725392.840363575,
     330.00000000000114, 986.36389129776705, 0.00050295446547331095, 0.64441680500000098, 4179.005918456779, 0.00015420337218329924},
    {340, 50000, -15683576.923673673,
     340, 983.50597349484815, 0.00043018557800759852, 0.65378735999999982, 4184.6658637801202, 0.00015623406534289366},
    {350, 50000, -15641691.44649804,
     350.00000020051755, 980.56988705504807, 0.00037300059309999356, 0.66192687515094195, 4192.8428047935549, 0.00015787066340626466},
    {360, 50000, -15599713.317036513,
     360.00000000000614, 977.55072743231278, 0.00032743476022934284, 0.6688462400000037, 4203.1419825700195, 0.00015913006098143665},
    {370, 50000, -15557622.766955297,
     370.0000000000183, 974.44309696472499, 0.00029066832134325382, 0.67455634500000916, 4215.2934699416692, 0.00016002595069836113},
    {380, 50000, -15515402.103085814,
     380.00000000002228, 971.24103603493529, 0.00026066121092892778, 0.67906808000000851, 4229.152169636377, 0.000160568372279306},
    {390, 50000, -15473034.459124567,
     390.00000000000307, 967.93794052901262, 0.00023591382489530994, 0.68239233500000063, 4244.6978152094789, 0.00016076346649574724},
    {400, 50000, -15430502.547334258,
     400.00000000007094, 964.52646388387416, 0.00021530736151405443, 0.68454000000001092, 4262.0349708576687, 0.00016061341698992625},
    {280, 100000, -15934677.177039757,
     279.99999999579609, 999.62487928559119, 0.0014300291330805663, 0.57871727999313261, 4211.0158121652175, 0.00013742937709264219},
    {290, 100000, -15892659.110051598,
     289.99999999873557, 997.09779658946127, 0.0011126737699047307, 0.5944063849980975, 4193.8003320028101, 0.00014173454574415329},
    {300, 100000, -15850780.487802632,
     299.99999999973136, 994.5114684213753, 0.00088614741034405799, 0.60880999999963015, 4182.9480988067153, 0.00014554567391675445},
    {310, 100000, -15808982.876832884,
     309.99999999998215, 991.86272142792097, 0.00072058248924853588, 0.62193901499997761, 4177.4402009991181, 0.00014888041122676718},
    {320, 100000, -15767217.408683423,
     320.00000000000904, 989.14811145249143, 0.00059698641565931977, 0.63380432000000997, 4176.3825576463469, 0.00015175916268484716},
    {330, 100000, -15725443.531594714,
     330.00000000000199, 986.36389129776683, 0.00050295446547330422, 0.64441680500000187, 4179.0059184567826, 0.00015420337218329932},
    {340, 100000, -15683627.762205768,
     340, 983.50597349484815, 0.00043018557800759852, 0.65378735999999982, 4184.6658637801202, 0.00015623406534289366},
    {350, 100000, -15641742.437254189,
     350.00000030667155, 980.56988702344847, 0.0003730005925604681, 0.6619268752308507, 4192.8428048922096, 0.00015787066342160845},
    {360, 100000, -15599764.465277312,
     360.00000000001495, 977.55072743231028, 0.00032743476022930793, 0.66884624000000925, 4203.1419825700323, 0.00015913006098143749},
    {370, 100000, -15557674.078314736,
     370.00000000004053, 974.44309696471805, 0.00029066832134318253, 0.67455634500002037, 4215.2934699416946, 0.00016002595069836283},
    {380, 100000, -15515453.583612423,
     380.00000000005002, 971.24103603492631, 0.00026066121092885276, 0.67906808000001939, 4229.1521696364134, 0.00016056837227930719},
    {390, 100000, -15473086.115328861,
     390.00000000000682, 967.93794052901137, 0.00023591382489530113, 0.68239233500000163, 4244.6978152094853, 0.00016076346649574724},
    {400, 100000, -15430554.386243684,
     400.00000000013051, 964.52646388385324, 0.00021530736151394311, 0.68454000000002013, 4262.0349708577814, 0.00016061341698992416},
    {280, 200000, -15934777.214565905,
     279.99999999182694, 999.62487928658254, 0.0014300291332301218, 0.57871727998664879, 4211.0158121734639, 0.00013742937709083333},
    {290, 200000, -15892759.401116669,
     289.99999999756244, 997.09779658976129, 0.0011126737699359091, 0.59440638499633269, 4193.8003320044345, 0.00014173454574367757},
    {300, 200000, -15850881.039684812,
     299.99999999952007, 994.51146842143055, 0.0008861474103481192, 0.60880999999933927, 4182.9480988068808, 0.00014554567391667915},
    {310, 200000, -15809083.697236596,
     309.99999999999636, 991.86272142791745, 0.00072058248924833238, 0.62193901499999527, 4177.4402009991145, 0.00014888041122677152},
    {320, 200000, -15767318.505777823,
     320.00000000003308, 989.14811145248473, 0.00059698641565906476, 0.63380432000003706, 4176.3825576463496, 0.00015175916268485353},
    {330, 200000, -15725544.91405699,
     330.00000000000813, 986.36389129776535, 0.00050295446547325424, 0.64441680500000798, 4179.0059184567863, 0.00015420337218330065},
    {340, 200000, -15683729.43926996,
     340, 983.50597349484815, 0.00043018557800759852, 0.65378735999999982, 4184.6658637801202, 0.00015623406534289366},
    {350, 200000, -15641844.418766484,
     350.00000051739556, 980.56988696072074, 0.00037300059148946501, 0.66192687538947581, 4192.8428050880475, 0.00015787066345206703},
    {360, 200000, -15599866.76175891,
     360.00000000004053, 977.55072743230232, 0.00032743476022920384, 0.66884624000002546, 4203.1419825700586, 0.00015913006098144033},
    {370, 200000, -15557776.701033611,
     370.00000000010044, 974.44309696469929, 0.00029066832134298169, 0.67455634500005113, 4215.2934699417756, 0.00016002595069836704},
    {380, 200000, -15515556.544665644,
     380.00000000012074, 971.24103603490357, 0.00026066121092866156, 0.67906808000004704, 4229.152169636518, 0.00016056837227930977},
    {390, 200000, -15473189.427737448,
     390.00000000002018, 967.93794052900705, 0.00023591382489527096, 0.68239233500000529, 4244.6978152095071, 0.0001607634664957473},
    {400, 200000, -15430658.064062536,
     400.00000000021885, 964.52646388382277, 0.00021530736151377636, 0.68454000000003412, 4262.0349708579397, 0.00016061341698992148},
    {280, 500000, -15935077.327144351,
     279.99999998060042, 999.62487928938765, 0.0014300291336531306, 0.57871727996830968, 4211.0158121967761, 0.0001374293770857175},
    {290, 500000, -15893060.274311883,
     289.99999999454707, 997.0977965905322, 0.0011126737700160425, 0.5944063849917961, 4193.8003320086136, 0.00014173454574245459},
    {300, 500000, -15851182.695331354,
     299.99999999922113, 994.51146842150888, 0.00088614741035386471, 0.60880999999892782, 4182.94809880712, 0.00014554567391657247},
    {310, 500000, -15809386.158447728,
     310.00000000023419, 991.86272142785367, 0.00072058248924495249, 0.62193901500029236, 4177.4402009990372, 0.00014888041122684541},
    {320, 500000, -15767621.797061026,
     320.00000000019241, 989.14811145244096, 0.00059698641565735899, 0.63380432000021603, 4176.3825576463678, 0.00015175916268489573},
    {330, 500000, -15725849.061443819,
     330.00000000004746, 986.36389129775375, 0.00050295446547292855, 0.64441680500004728, 4179.005918456799, 0.00015420337218330959},
    {340, 500000, -15684034.470462536,
     340, 983.50597349484815, 0.00043018557800759852, 0.65378735999999982, 4184.6658637801202, 0.00015623406534289366},
    {350, 500000, -15642150.363303373,
     350.00000113689094, 980.56988677631148, 0.0003730005883408833, 0.66192687585580867, 4192.8428056637686, 0.0001578706635416109},
    {360, 500000, -15600173.651203705,
     360.00000000018173, 977.55072743225901, 0.00032743476022862683, 0.66884624000011461, 4203.1419825702214, 0.00015913006098145538},
    {370, 500000, -15558084.569190238,
     370.00000000039347, 974.44309696460698, 0.00029066832134201409, 0.6745563450002009, 4215.293469942153, 0.00016002595069838826},
    {380, 500000, -15515865.427825302,
     380.00000000045861, 971.24103603479352, 0.00026066121092774476, 0.67906808000017937, 4229.1521696370173, 0.0001605683722793221},
    {390, 500000, -15473499.364963213,
     390.00000000009248, 967.93794052898261, 0.00023591382489510793, 0.68239233500002516, 4244.6978152096272, 0.00016076346649574743},
    {400, 500000, -15430969.09751909,
     400.00000000022771, 964.52646388381982, 0.00021530736151375877, 0.68454000000003545, 4262.0349708579542, 0.00016061341698992126},
};
int failures = 0;
// TOLERANCE. Both sides are now evaluated at the SAME temperature -- OpenFOAM's own Trec, from the CORR
// block -- so there is no propagated inversion error left to allow for: the only difference either side
// can have is the arithmetic of the correlations themselves. Measured 2026-09-15: worst 2.36e-16
// relative over all 52 cells x 6 fields -- one ulp. 1e-13 is a floor for double-precision
// reassociation, not a budget, and it stays eleven orders below anything a WIRING error produces --
// wrong p, wrong index or a one-iteration lag are all >= 1e-2 here, which is what this test exists
// to catch. Mutation-verified below rather than asserted.
double worstRel = 0;

void expect(const char* what, int i, double got, double want, double tol = 1e-13)
{
    const double rel = std::fabs(got - want)/std::fmax(std::fabs(want), 1e-300);
    worstRel = std::fmax(worstRel, rel);
    if (!(rel <= tol))
    {
        std::printf("  FAIL %-10s cell %d  brae %.17g  OF %.17g  rel %.3e\n", what, i, got, want, rel);
        ++failures;
    }
}
}   // namespace

int main()
{
    const int n = static_cast<int>(kOF.size());
    ThermoCoeffs c;
    c.model          = ThermoModel::liquidH2O;
    c.internalEnergy = true;            // squareBendLiq: energy sensibleInternalEnergy
    c.rhoThermoType  = true;            // heRhoThermo

    std::vector<scalar> he(n), p(n), Tbad(n);
    for (int i = 0; i < n; ++i)
    {
        he[i]   = kOF[i].Es;
        p[i]    = kOF[i].p;
        // Deliberately wrong guess, reversed within each 13-point pressure block.
        Tbad[i] = kOF[(i/13)*13 + (12 - i%13)].T;
    }

    DeviceThermo th;
    th.allocate(n);
    th.he.copyFrom(he);
    th.T.copyFrom(Tbad);
    // Solver-side rho seeded with a sentinel: deviceThermoCorrect must NOT touch it. OF's calculate()
    // writes the thermo's rho_, and the solver's own rho is assigned later by `rho = thermo.rho()`.
    std::vector<scalar> sentinel(n, -12345.0);
    th.rho.copyFrom(sentinel);

    DeviceBuffer<scalar> pD;
    pD.copyFrom(p);

    deviceThermoCorrect(th, pD, c);     // <-- the real entry point

    const std::vector<scalar> T  = th.T.host(),  Cp = th.CpField.host();
    const std::vector<scalar> mu = th.mu.host(), ka = th.kappa.host();
    const std::vector<scalar> rt = th.rhoThermo.host(), al = th.alpha.host();
    const std::vector<scalar> rs = th.rho.host();

    std::printf("integration: deviceThermoCorrect() liquid path, %d cells over 4 pressures\n", n);
    for (int i = 0; i < n; ++i)
    {
        const Row& r = kOF[i];
        // r.Trec, not r.T: OpenFOAM's answer for this cell's guess, which is what calculate() writes.
        expect("T",         i, T[i],  r.Trec);
        expect("Cp",        i, Cp[i], r.Cp);
        expect("mu",        i, mu[i], r.mu);
        expect("kappa",     i, ka[i], r.kappa);
        expect("rhoThermo", i, rt[i], r.rho);
        // OF's own alphah(p, Trec), not kappa/Cp recomposed here -- heRhoThermo.C:91 writes what the
        // mixture returns, and asserting a recomposition would hide a selector that stopped agreeing.
        expect("alpha",     i, al[i], r.alphah);
    }
    std::printf("  T, Cp, mu, kappa, rhoThermo, alpha all match OF, worst %.2e relative\n", worstRel);

    // The two-density distinction must survive integration: correct() writes rhoThermo, never rho.
    {
        int touched = 0;
        for (int i = 0; i < n; ++i) if (rs[i] != scalar(-12345.0)) ++touched;
        if (touched)
        {
            std::printf("  FAIL deviceThermoCorrect wrote the SOLVER rho in %d cells -- it must write\n"
                        "       rhoThermo only; the solver assigns rho = thermo.rho() after the pressure solve\n",
                        touched);
            ++failures;
        }
        else std::printf("  solver rho untouched (rhoThermo is what correct() writes)\n");
    }

    // Properties must come from the NEW T, not the guess. If they lagged, evaluating at Tbad would match.
    {
        int lagged = 0;
        for (int i = 0; i < n; ++i)
            if (std::fabs(Cp[i] - H2OLiquid::Cp(Tbad[i])) < 1e-9 && std::fabs(Tbad[i] - kOF[i].T) > 1.0)
                ++lagged;
        if (lagged)
        {
            std::printf("  FAIL %d cells carry properties evaluated at the PREVIOUS temperature --\n"
                        "       the property update ran before the inversion\n", lagged);
            ++failures;
        }
        else std::printf("  properties evaluated at the updated T, not the guess\n");
    }

    // The gas path must still take its own branch: same call, perfectGas coefficients, liquid fields
    // must stay unallocated.
    {
        ThermoCoeffs g;                 // default perfectGas
        DeviceThermo gt;
        gt.allocate(8);
        std::vector<scalar> ghe(8, 3.0e5), gp(8, 1e5);
        gt.he.copyFrom(ghe);
        DeviceBuffer<scalar> gpD;
        gpD.copyFrom(gp);
        deviceThermoCorrect(gt, gpD, g);
        if (gt.CpField.size() != 0 || gt.kappa.size() != 0)
        {
            std::printf("  FAIL the perfectGas branch allocated liquid fields\n");
            ++failures;
        }
        else std::printf("  perfectGas branch unchanged (no liquid fields, closed-form he->T)\n");
    }

    std::printf("test_liquid_correct: %d failures\n", failures);
    return failures ? 1 : 0;
}
