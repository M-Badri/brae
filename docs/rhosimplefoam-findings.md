# rhoSimpleFoam findings

A source-level audit of brae against the six stock OpenFOAM v2412 rhoSimpleFoam tutorials, plus direct matrix
comparison against OpenFOAM. **48 findings, 42 closed.** All six tutorials now run as shipped.

Full detail, measurement by measurement, is in the solver's `PORT.md`. This page is the index.

## The pattern worth remembering

The dangerous class is not "missing feature". It is **present, parsed, and never applied** — that converges to a
plausible wrong answer and says nothing. Sixteen of the forty-eight were that shape.

Every fix had to either apply the input or refuse the case. Nothing was allowed to stay silent.

## A. Silent wrong behaviour — ran, converged, wrong

| # | What | Cost when measured |
|---|---|---|
| A1 | `inletOutlet` on T never resolved from the flux sign | **T off by 276%**, all 6 tutorials |
| A3 | Turbulent-inlet BCs frozen at set-up, not recomputed each iteration | **k off by a factor of 44,000** |
| A4 | `div(phi,K)` / `div(phi,Ekp)` ignored `linearUpwind`, ran upwind | T 2.0e-02 → 1.9e-06 (**10,600×**) |
| A6 | Per-face `Prt` matched by exact patch name only, so a regex key fell back to the model default | T 5.3e-03 → 2.7e-07 (**19,000×**) |
| A13 | Duplicate keys resolved to the **first** entry; OpenFOAM takes the **last** | read `endTime 0.4` not `1200` — **ran zero iterations and reported success** |
| A14 | Repeated sub-dictionaries replaced wholesale; OpenFOAM **merges** them | refused a case OpenFOAM runs |
| A8 | `pMin`/`pMax` reference scanned every patch, not only those fixing a value | pMax **9× too large**, so the limiter did nothing |
| A12 | `grad(K)` took its boundary from the *energy* field's descriptor | fatal under linearUpwind; masked under limitedLinear |
| A2 | `fvOptions` never read by the compressible driver | angledDuct silently dropped its porosity source |
| A5 | `uniformFixedValue` with an expression reused a stale scalar | squareBendLiq T walls ran at a constant 350 K |
| A7 | `freestream` valueFraction mixed numerator and denominator | small at a true far field |
| A9 | `tangentialVelocity` on `pressureInletOutletVelocity` silently ignored | — |
| A10 | `overset` treated as a constraint patch type | an overset case **ran, converging to a wrong answer** |
| A11 | Written `boundaryField` was a pass-through of the input, not the computed value | broke post-processing and restarts |
| A15 | A bare value entry (`inletValue (0 0 0);`) threw where OpenFOAM tolerates it | failed on a file OF accepts |
| A16 | Nothing brae wrote could be read back by OpenFOAM | two independent writer defects |

All sixteen are fixed or refused, each with a gate and a mutation test.

## B. Honest refusals — limited scope, never wrong

| # | What | Now |
|---|---|---|
| B1 | `transonic yes` | ✅ implemented |
| B2 | liquid / non-perfectGas thermo | ✅ implemented (NSRDS + OpenFOAM's own `he→T` inversion) |
| B3 | `nutUWallFunction` | ✅ implemented (STEPWISE blender; others still refused) |
| B4 | `coded` U, `expression` T, `functionObjectTrigger` | ✅ implemented |
| B5 | temperature-gradient BCs | ✅ `fixedGradient` and `mixed`; `externalWallHeatFluxTemperature` still refused, by name |
| B6 | `alphatJayatillekeWallFunction`, isentropic `totalPressure`, `flowRateInletVelocity extrapolateProfile` | ☐ backlog |
| B7 | Spalart-Allmaras and realizableKE, compressible (ρ-weighting) | ☐ backlog |

## C. No effect then, but silent

Inputs the case supplied that brae read and ignored, or resolved by the wrong rule. All closed.

| # | What |
|---|---|
| C1 | `interpolationSchemes` never parsed — a non-`linear` entry ran linear |
| C2 | `grad(k)` / `grad(omega)` `cellLimited` parsed by nothing |
| C3 | `limited` on a `div` line wrongly set the laplacian's non-orthogonal flag |
| C4 | `div(phi,{h,e,K,Ekp})` flags OR-accumulated into one slot |
| C5 | No `residualControl` — always ran to `endTime` |
| C6 | `startFrom latestTime` ignored, hardcoded to `0/` |
| C7 | Schemes resolved by hardcoded per-field flags, not by OpenFOAM's name lookup |

## E. Found by the dictionary audit

Entries read off disk and never used — the `[unread]` notice exists because of these. E1–E4 and E7 closed;
E5 (the audit does not run on a refused case) and E6 (`thermoRR` not user-overridable) remain.

## F. aerofoilNACA0012 — six the ducts could not expose

| # | What |
|---|---|
| F1 | `alphat` never set by brae's `validate()` |
| F2 | `he` measured about the wrong reference temperature |
| F3 | `bound(field, SMALL)` applied to `he`, which OpenFOAM does not bound |
| F4 | `linearUpwind`'s gradient **argument** ignored |
| F5 | `thermo.correct()` also overwrote `rho` |
| F6 | The boundary half of F5 |

## D. Why these survived — the verification gaps

The more useful half of the audit. Each defect above existed because something was not being checked.

| # | Gap | Now |
|---|---|---|
| D1 | No gate case used `inletOutlet` on T, so A1 was invisible to all 9 compressible gates | ✅ `validation/rhoIO` |
| D2 | Boundary coefficients never compared against OpenFOAM for real BC types | ✅ four BC types at once, every patch to **machine precision** |
| D4 | No gate used a turbulent inlet whose set-up value differs from the converged one | ✅ `validation/rhoTI` |
| D5 | One gate looked dead — **it was 28.** Every `*_vs_openfoam.sh` predating the `cf`→`brae` rename was unregistered, pointed at a dead path, **and printed its comparison without asserting anything.** `validation/` advertised 28 OpenFOAM gates that did not exist | ✅ quarantined, and a `gate_registration` guard now fails the build |
| D3 | Only 2 of 6 tutorials started | ✅ all six run |

**D5 is the one to take away.** The thing meant to catch silent gaps had one of its own, one level up.

## Two notes worth keeping

**brae's gas constant was more correct than OpenFOAM's, and that made it wrong.** brae used CODATA-2018
`RR = 8314.46261815324`; OpenFOAM v2412 computes `8314.47006650545` from its own rounded constants — the pre-2019
value. The gap is 8.958e-07 relative and it reaches `R`, `psi` and `rho` in every compressible case. Found as an
unexplained 9.06e-07 floor on `rho` that survived after brae started stopping at OpenFOAM's own iteration count.
The contract is to reproduce OpenFOAM, not to be right about physics.

**The transport operator was never the problem.** Matrix assembly checks out against OpenFOAM's own
`fvScalarMatrix` at 1.76e-06 on the diagonal and 6.66e-07 on the source. Every defect above was an input reaching
the operator wrongly, or an output leaving it wrongly.
