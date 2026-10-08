# Calibrated Transfer Between Two Simulated Experimental Systems

A small, fully reproducible test bench for one question:

> When a model learned by an autonomous experimentation loop on one system is moved to a second
> system, can its uncertainty estimates still be trusted, and how many experiments on the new
> system does it take to fix them?

The project is motivated by the problem setting of sharing knowledge between autonomous
material-exploration systems (for example, Yoshida et al., *Networking autonomous material
exploration systems through transfer learning*, npj Computational Materials, 2025). **It is not a
reproduction of that work or of any published method.** It is an independent, simplified
experiment on synthetic data that isolates the calibration question.

## What it does

1. **Two synthetic systems** measure the same property landscape `f(x)` over two normalised
   process parameters.
   - **System A (source):** low noise, no distortion.
   - **System B (target):** different gain (1.15) and offset (+0.20), larger input-dependent
     noise, and measurements drawn from a shifted input region (covariate shift).
2. **Autonomous exploration on A:** a Gaussian-process loop that always runs the next experiment
   where the model is least certain (60 experiments).
3. **Split conformal prediction** on fresh A measurements gives 90% prediction intervals.
4. **Transfer to B**, comparing three ways of building intervals:
   - `naive`: reuse A's interval half-width on B (no target data).
   - `recalibrated`: recompute the conformal quantile from *n* target measurements.
   - `affine + conformal`: fit `y ≈ a·μ + b` on half of the *n* target measurements (absorbing
     gain and offset), then calibrate conformally on the other half.
5. Repeat over **30 random seeds** and report marginal coverage, coverage in the high-noise
   region (`x1 > 0.75`), and interval width.

## Results (30 seeds, target coverage 90%)

| Method | Target measurements | Coverage (mean ± sd) | Coverage, high-noise region | Mean width |
|---|---:|---:|---:|---:|
| A, in-distribution | 0 | 0.900 ± 0.023 | 0.899 | 0.222 |
| B, naive (reuse A quantile) | 0 | **0.213 ± 0.029** | 0.201 | 0.222 |
| B, recalibrated | 20 | 0.911 ± 0.055 | 0.875 | 1.187 |
| B, recalibrated | 40 | 0.898 ± 0.040 | 0.858 | 1.117 |
| B, recalibrated | 80 | 0.901 ± 0.037 | 0.861 | 1.119 |
| B, recalibrated | 160 | 0.900 ± 0.025 | 0.858 | 1.111 |
| B, affine + conformal | 20 | 0.910 ± 0.105 | 0.878 | 0.936 |
| B, affine + conformal | 40 | 0.923 ± 0.044 | 0.882 | 0.857 |
| B, affine + conformal | 80 | 0.893 ± 0.050 | 0.843 | 0.758 |
| B, affine + conformal | 160 | 0.902 ± 0.032 | 0.853 | 0.759 |

![Coverage and width versus number of target measurements](results/coverage_vs_target_samples.png)

**What the numbers show**

- Calibration on the source system works as designed: coverage on A is 0.900.
- Reusing A's intervals on B fails badly (coverage ≈ 0.21 against a 0.90 target). The size of
  this gap depends on how strongly I chose to distort B, so read it as "a shift of this size
  breaks naive reuse", not as a general rate of failure.
- Roughly 20 target measurements are enough to restore marginal coverage to about 0.90.
- Correcting gain and offset first gives intervals about 32% narrower at similar coverage
  (width 0.76 versus 1.11 at 80 to 160 target measurements): the same coverage is reached with a
  more informative interval.
- **Marginal coverage hides a local gap.** In the high-noise region coverage is 0.84 to 0.88
  even when overall coverage is on target. Conformal guarantees are marginal, so a method can
  look calibrated on average and still under-cover where noise is highest. This is the same
  kind of marginal-versus-conditional gap I found for rapid ramp events in my sky-image solar
  forecasting work.

## Limitations (please read)

- Everything is **synthetic**. There is no real instrument data, and the shifts used (affine
  distortion, heteroscedastic noise, covariate shift) are simple on purpose.
- The conformal guarantee assumes the target calibration points and target test points are
  exchangeable. Here they are drawn from the same distribution; real transfer settings may not
  satisfy that.
- The base model is a plain Gaussian process, and the exploration loop is basic uncertainty
  sampling, not a state-of-the-art autonomous experimentation system.
- Distortion strengths and noise levels were chosen by me, so absolute numbers (especially the
  naive coverage) are specific to this setup.

## Possible next steps

- Weighted conformal prediction for covariate shift, and locally adaptive (normalised) scores to
  close the high-noise-region gap.
- Choose the target experiments actively instead of sampling them at random.
- Time-varying drift in the target system, and tests on real or published materials datasets.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q                 # 10 tests, about 5 seconds
python run_experiment.py            # 30 seeds, about 45 seconds
python run_experiment.py --seeds 5  # quick check
```

Outputs are written to `results/` (`raw_results.csv`, `summary.csv`, the figure above). All
randomness is seeded, so reruns reproduce the table.

## Repository layout

```
tcdemo/
  systems.py      synthetic landscape and the two measurement systems
  explore.py      Gaussian-process uncertainty-sampling loop
  conformal.py    split conformal quantile, coverage, width, affine corrector
  experiment.py   the transfer experiment and summary statistics
run_experiment.py script that runs everything and makes the figure
tests/            unit tests (conformal validity, simulators, end-to-end behaviour)
```

## Author

Maliha Ehsan, BS Computer Science, University of Management and Technology (UMT), Lahore.
MIT licensed.
