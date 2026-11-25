# Simulation for the actuator FDI paper

`pip install numpy numba matplotlib` then `python run_experiments.py` (about 70 s).

- `actuator_sim.py`: plant (SPM PMSM, inverter with dead time, gearbox, friction, obstacle), 20 kHz FOC + PD position loop, back-EMF residual, momentum observer, following-error baseline, isolation logic.
- `run_experiments.py`: threshold calibration, 1 h false-alarm runs, robustness sweep, 6x50 Monte Carlo fault trials, ROC, speed sweeps, and all figures in `../figures/`.
- `results/results.json`: every number quoted in the paper.

Parameters in `P` are representative, not identified from the testbed. Replace them and rerun before submission.
