/*---------------------------------------------------------------------------*\
  liqref -- print OpenFOAM's OWN liquid thermo answers, as an oracle for brae.

  WHY THIS EXISTS. brae's h2o correlations are already gated against a table this tool's ancestor
  produced. What that table cannot gate is the he -> T INVERSION's PATH. OpenFOAM's
  species::thermo<>::T (thermoI.H:43-88) is a do-while that stops when the TEMPERATURE STEP falls below
  T0*tol_ with tol_ = 1e-4 (thermo.C:33) -- it does NOT iterate to convergence, so the temperature it
  returns DEPENDS ON THE INITIAL GUESS T0. A table of (Es, p) -> T recorded at one T0 therefore cannot
  tell a path-faithful port from a port that simply converges harder.

  So this prints TEs / THs for a grid of (energy, p, T0) triples, with T0 deliberately cold and hot
  relative to the answer, at 17 significant digits. It also prints the raw properties so the same table
  covers the correlations and the inversion together.

  Nothing here is brae's arithmetic: the mixture is constructed exactly as liquidThermo.H builds it for
  `properties liquid` + `energy sensibleInternalEnergy`, and every number comes from OpenFOAM calling
  itself.
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "thermophysicalPropertiesSelector.H"
#include "liquidProperties.H"
#include "sensibleInternalEnergy.H"
#include "sensibleEnthalpy.H"
#include "thermo.H"
#include "IOmanip.H"
#include <vector>

using namespace Foam;

typedef species::thermo
<
    thermophysicalPropertiesSelector<liquidProperties>,
    sensibleInternalEnergy
> eThermo;

typedef species::thermo
<
    thermophysicalPropertiesSelector<liquidProperties>,
    sensibleEnthalpy
> hThermo;

int main(int argc, char *argv[])
{
    argList::noBanner();
    argList::noParallel();
    argList::noFunctionObjects();
    argList::addNote("print OpenFOAM's own H2O liquid properties and he->T inversions");
    argList args(argc, argv, false, false, false);

    const eThermo eT(thermophysicalPropertiesSelector<liquidProperties>("H2O"));
    const hThermo hT(thermophysicalPropertiesSelector<liquidProperties>("H2O"));

    Info<< setprecision(17);

    // The pressures and temperatures the squareBendLiq tutorial actually visits, plus the ends of the
    // correlation range so a port cannot pass by being right only in the middle.
    // 5.0e4 added 2026-09-15: tests/test_liquid_correct.cu and test_etot.cu are built on p = 50000 and
    // had no INV row to compare against, which is how they ended up asserting the TRUE temperature
    // instead of OpenFOAM's inverted one.
    const std::vector<scalar> ps{5.0e4, 1.0e5, 2.0e5, 5.0e5, 1.0e6};
    // 310/340/370 added 2026-09-15 so the grids the he->T tests are built on are covered.
    const std::vector<scalar> Ts{280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0,
                                 370.0, 380.0, 390.0, 400.0, 450.0, 500.0, 550.0, 600.0, 640.0};
    // T0 relative to the answer: exact, warm, cold, hot, and very cold -- the spread that exposes a
    // stopping test measured against T0 rather than against convergence.
    const std::vector<scalar> dT0{0.0, 0.1, -50.0, +50.0, -150.0, +150.0};

    Info<< "# PROPS p T rho mu kappa Cp Cv Es Hs" << nl;
    for (const scalar p : ps)
    {
        for (const scalar T : Ts)
        {
            Info<< "PROPS " << p << ' ' << T
                << ' ' << eT.rho(p, T)
                << ' ' << eT.mu(p, T)
                << ' ' << eT.kappa(p, T)
                << ' ' << eT.Cp(p, T)
                << ' ' << eT.Cv(p, T)
                << ' ' << eT.Es(p, T)
                << ' ' << hT.Hs(p, T)
                << nl;
        }
    }

    // THE POINT OF THE FILE. For each (p, T) the energy is evaluated once, then inverted from several
    // starting guesses. A path-faithful port reproduces EVERY row; a port that iterates to convergence
    // reproduces only the rows whose T0 happens to be close enough that OF converged too.
    // Every (Ttrue, T0) PAIR from the grid, not just fixed offsets -- a test that reverses the list
    // (tests/test_hetot.cu) needs the pair (280 from 400), which no offset grid contains. This is the
    // table a fixture should be built from: the right-hand side of "did brae reproduce OpenFOAM's
    // inversion", as opposed to "did brae recover the true temperature", which OpenFOAM itself does not.
    Info<< "# INVX form p Ttrue T0 Trecovered" << nl;
    for (const scalar p : ps)
        for (const scalar T : Ts)
        {
            const scalar es = eT.Es(p, T);
            const scalar hs = hT.Hs(p, T);
            for (const scalar t0 : Ts)
            {
                Info<< "INVX e " << p << ' ' << T << ' ' << t0 << ' ' << eT.TEs(es, p, t0) << nl;
                Info<< "INVX h " << p << ' ' << T << ' ' << t0 << ' ' << hT.THs(hs, p, t0) << nl;
            }
        }

    Info<< "# INV form p Ttrue T0 Trecovered" << nl;
    for (const scalar p : ps)
    {
        for (const scalar T : Ts)
        {
            const scalar es = eT.Es(p, T);
            const scalar hs = hT.Hs(p, T);
            for (const scalar d : dT0)
            {
                const scalar T0 = T + d;
                if (T0 <= 0) continue;
                Info<< "INV e " << p << ' ' << T << ' ' << T0 << ' ' << eT.TEs(es, p, T0) << nl;
                Info<< "INV h " << p << ' ' << T << ' ' << T0 << ' ' << hT.THs(hs, p, T0) << nl;
            }
        }
    }

    // OUT-OF-RANGE GUESSES. tests/test_etot.cu and test_hetot.cu hand the inversion a guess from well
    // outside [Tt, Tc] -- what a diverging outer iteration would produce -- and used to assert that brae
    // PROJECTED it back and still recovered the answer. OpenFOAM does no such thing: limit() is the
    // identity for a liquid (liquidPropertiesI.H:28-31), so the first Newton step is taken at the bad
    // guess itself. This block records what OpenFOAM actually returns there. Added 2026-09-15.
    //
    // T0 = -500 IS NOT IN THE LIST, AND THAT IS THE MEASUREMENT. thermoI.H:55-60 guards `if (T0 < 0)`
    // and raises FatalErrorInFunction << ... << abort(FatalError) -- OpenFOAM REFUSES a negative initial
    // temperature rather than inverting from it. abort() is not routed through throwExceptions(), so the
    // process core-dumps and takes the rest of the table with it; observed 2026-09-15, which is how the
    // guard was found. A port must refuse it too, so there is nothing for this table to carry.
    Info<< "# OOR form p Ttrue T0 Trecovered" << nl;
    {
        const scalar p = 1.0e5;
        const scalar T = 300.0;
        const scalar es = eT.Es(p, T);
        const scalar hs = hT.Hs(p, T);
        for (const scalar t0 : {0.0, 5000.0})
        {
            Info<< "OOR e " << p << ' ' << T << ' ' << t0 << ' ' << eT.TEs(es, p, t0) << nl;
            Info<< "OOR h " << p << ' ' << T << ' ' << t0 << ' ' << hT.THs(hs, p, t0) << nl;
        }
    }

    // THE INTEGRATION ORACLE. heRhoThermo::calculate() (heRhoThermo.C:79-91) inverts he -> T and then
    // evaluates psi, rho, mu and alphah AT THE RECOVERED T -- not at the true one, which it never has.
    // tests/test_liquid_correct.cu compares deviceThermoCorrect()'s whole output against OpenFOAM, so
    // its right-hand side has to be that same composition: OF's own inversion followed by OF's own
    // properties at OF's own answer. Reading properties off the PROPS grid instead compares brae at
    // T_recovered against OpenFOAM at T_true, and mu -- exp(a + b/T + ...) with b = 3670.6, so
    // |dln(mu)/dT| ~ 0.047 /K -- turns the inversion's 1.1e-06 K path difference into 5e-08 relative,
    // fifty times the tolerance that test is trying to hold. Added 2026-09-15.
    // The guess is the reversed one that fixture uses: 680 - T within each pressure block.
    Info<< "# CORR p Ttrue T0 Trec rho mu kappa Cp alphah" << nl;
    for (const scalar p : {5.0e4, 1.0e5, 2.0e5, 5.0e5})
    {
        for (const scalar T : {280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0,
                               370.0, 380.0, 390.0, 400.0})
        {
            const scalar T0 = 680.0 - T;
            const scalar Trec = eT.TEs(eT.Es(p, T), p, T0);
            Info<< "CORR " << p << ' ' << T << ' ' << T0 << ' ' << Trec
                << ' ' << eT.rho(p, Trec)
                << ' ' << eT.mu(p, Trec)
                << ' ' << eT.kappa(p, Trec)
                << ' ' << eT.Cp(p, Trec)
                << ' ' << eT.alphah(p, Trec)
                << nl;
        }
    }

    Info<< "# END" << nl;
    return 0;
}
