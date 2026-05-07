# Push-Heavy Benchmark

This directory is reserved for the auxiliary push-heavy evaluation added for BEACON.

The benchmark harness is implemented in:

- `/Users/ishita/Documents/GitHub/BEACON-semantics/experiments/run_push_heavy_planner_evaluation.py`

Its purpose is to evaluate planners in structured scenes where movable obstacle plugs compete against substantially longer rigid-barrier detours, so that BEACON's push-versus-reroute policy can be exercised more directly than in the main semantics sweep.

The main semantics sweep should still be treated as the primary reported result for the current paper figures:

- BEACON matches D* Lite on success.
- BEACON does not reduce traversal energy relative to D* Lite in that sweep.
- A tuned BEACON pass narrows the energy gap but does not invert it.
