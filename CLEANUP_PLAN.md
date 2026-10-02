# Cleanup And Implementation Plan

## Terminology Cleanup Plan

1. Rename process-oriented orchestration and analysis entry points to neutral
   validation terminology.
2. Replace process-oriented notebook metadata, output roots, and prose while
   preserving checkpoint compatibility through explicit version changes.
3. Move existing output directories without deleting generated evidence and
   update every source, test, notebook, and documentation reference.
4. Verify that source-controlled project files contain no submission-process
   terminology, then run lint, typing, and the full test suite.

## Scope

This pass is restricted to `calcium-transient-rising-flank/`. It aligns the
testable library with the supplied analysis source: `src/core/causalised-GC.py`
is the active c-GC/c-GC* implementation, with event-aware extensions that keep
the old rising-flank selected-frame logic available while adding a physical
time mode for discontinuous event segments.

## Baseline Evidence

Before production edits, the supplied estimator source could not be used
through the public package surface:

- `src/core/rising_flanks.py` imported undeclared `sklearn` before its public class
  can be constructed.
- The documented legacy `from rising_flanks` and `from others_funcs` imports
  had no top-level forwarding modules under `src/`.

Inspection also identified calls in `RisingFlanks.fit_rising` whose signatures
do not match the methods defined in the same class, notebook-specific plotting
state, unused imports, and mutation of input arrays while plotting. The
notebooks nevertheless establish the intended contract: raw traces and
per-ROI rising/falling frame indices are supplied to the core c-GC analysis.
Repairs will preserve that contract rather than substitute a new estimator.

Inspection of the typed package found package-owned Granger substitutions that
must remain out of the estimator path. The retained public adapter is
`CausalisedGC`, which delegates to modified supplied c-GC/c-GC* code rather
than implementing a separate Granger learner.

## Manuscript-Derived Deliverables

1. Preprocessing scenarios A-D: explicit bad-ROI exclusion, declared artifact
   interpolation, and smoothing.
2. Fixed-length signal representations: full, AR(1)-deconvolved, rising, and
   falling traces, plus decay-null falling residuals.
3. Directed estimation: a thin result-shaping adapter around the supplied
   c-GC/c-GC* implementation. A later method-validation extension adds an
   explicitly labeled conditional VAR-Granger baseline; it is not used as a
   substitute inside the c-GC/c-GC* pipeline.
4. Metrics: `W_IC`, `W_IC_bin`, paired `Delta W_IC`, `W_RC`, edge-recovery,
   and graph-stability summaries.
5. Validation: synthetic event/calcium generation and cyclic-shift null
   support where it invokes the supplied core estimator.
6. Deployment orchestration: run all representations and paired
   rise-versus-fall analyses over scenarios A-D.
7. Visualization and compatibility wrappers for the original script module
   names.

## Cleanup Order

1. Update unit tests to require the core-backed estimator boundary and reject
   unsupported extension parameters.
2. Repair only the invocation/import defects needed to run
   `RisingFlanks.fit_rising` without introducing an alternative GC algorithm.
3. Reduce `estimators.py` to result shaping around `GcStar`; remove
   package-owned GC/MVGC learner implementations from the pipeline path.
4. Update package documentation and smoke examples to state the supported
   boundary.
5. Run imports, unit tests, compilation checks, and a smoke pipeline.

## Scientific Boundaries

- The supplied c-GC/c-GC* implementation is used as the computational
  foundation; it does not claim anatomical synapse recovery.
- Segment-aware physical event mode is implemented inside the supplied
  c-GC/c-GC* core path for discontinuity handling. Cross-representation
  fall-to-rise GC is supported on the complete physical time axis; combining
  it with selected-frame or segment filtering remains unsupported.
- The method-validation baseline module may expose conditional VAR-Granger,
  provided its predictive (not interventional) semantics remain explicit.
- Optional latent-confounding-aware algorithms such as LPCMCI and SVAR-FCI are
  represented by an adapter boundary only unless their dependencies and
  estimator configuration are explicitly supplied later.
- If c-GC returns only oriented adjacency, weighted GC metrics are not claimed;
  `W_IC_bin` is reported instead.


look at https://arxiv.org/pdf/2102.09403
