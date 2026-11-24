# Forecast validation and release policy

The original two-store LightGBM had 97.79% WAPE and −88.55% bias. In its fitted period, 67.18% of product-weeks had zero volume. The L1 objective estimates a conditional median, which can sit near zero for intermittent purchases. The model predicted 539 L against 4,707 L on the validation window. Underpredictions on positive weeks accounted for 4,385 L of absolute shortfall.

The statewide model uses a Tweedie objective to estimate expected volume with nonnegative, zero-inclusive targets. Calendar offsets mark Thanksgiving, Christmas, New Year and Independence Day. The comparison uses one-week-ahead features and a frozen fit within each chronological window. Later weeks can use earlier actuals once those weeks close. Three May–October development windows informed model selection and gate limits. The November–December holdout remained separate.

| Window start | Seasonal-naive WAPE | LightGBM WAPE | LightGBM bias |
| --- | ---: | ---: | ---: |
| May 19, 2025 | 32.32% | 28.08% | +1.96% |
| July 14, 2025 | 31.22% | 30.42% | −5.23% |
| September 8, 2025 | 37.86% | 28.22% | +2.69% |
| November 3, 2025 holdout | 29.89% | 30.99% | −8.00% |

The statewide policy requires at least six validation weeks and 100 paired observations. WAPE must be at most 40% and absolute bias at most 15%. Protected categories selected from the fitted periods of either model require measurable volume, at least six product-weeks, WAPE at most 50%, absolute bias at most 25% and WAPE within 10 percentage points of the incumbent. Challengers require at least 2% relative aggregate WAPE improvement on weeks after the incumbent's validation window. An initial baseline still faces absolute, sample and operational gates. Bundled sparse and CI snapshots use a looser, separately reported showcase profile. The statewide limits are engineering thresholds informed by the development windows.

Intervals use the 10th and 90th percentiles of errors from a frozen earlier fit on the last eight pre-validation weeks. Errors are scaled by recent purchase volume. The eight-week development interval coverage was 80.0–81.1%. On the six-week holiday holdout, coverage fell to 76.7% with a mean width of 1,728 L. The nominal 80% level is a calibration target. The worker records prospective shadow estimates from stored issuance-time features before the reporting week closes and before accepting a matching label.

## Reproduce from cached snapshots

```sh
uv run --locked python scripts/backtest.py --statewide /path/to/statewide --demo /path/to/two-store --output reports/forecast-validation.json
uv run --locked spiritops benchmark --mode full --snapshot-root /path/to/statewide --report reports/statewide-lifecycle.json
```

The lifecycle replay uses the September development window to demonstrate promotion, the later holdout to test rejection, and the final two December weeks to measure prospective shadow performance inside a historical clock. It records the temporary monitoring sensitivity change and the injected artifact failure.

[Forecast comparison](../reports/forecast-validation.json) · [Statewide lifecycle evidence](../reports/statewide-lifecycle.json)
