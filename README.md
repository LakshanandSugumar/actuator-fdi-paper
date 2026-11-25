# Isolating Sensor Faults from Mechanical Events in Robot Joint Actuators

IEEE conference paper draft (IEEEtran, targeting IEEE/ASME AIM) on model-based fault detection and isolation for PMSM robot joint actuators: a back-EMF voltage residual isolates encoder faults, and a momentum observer detects collisions and payload changes.

## Contents

- `main.tex`, `refs.bib`: paper source and bibliography
- `main.pdf`: compiled draft
- `figures/`: simulation figures used in the paper
- `sim/`: simulation code (`actuator_sim.py`, `run_experiments.py`) and `results/` (all numbers quoted in the paper)

## Build

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Reproduce the simulations

```
cd sim
pip install numpy numba matplotlib
python run_experiments.py   # about 70 s, regenerates figures/ and results/
```

## Status

Work-in-progress draft. Simulation parameters are representative, not identified from the testbed, and the hardware results section is still a placeholder. `refs.bib` entries still need verifying.
