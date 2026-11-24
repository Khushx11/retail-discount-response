# Experiments

The November expanded study is the result set used here. Its calendar splits are training before 5 November 2018, validation from 5 November 2018 through 5 May 2019 and historical test from 6 May 2019. The feature builder asserts that labels finish before or at the next partition's first forecast. Validation selects configurations. Each model family has eight candidate configurations and three final seeds. [Study plan](results/plan.json) · [Selections](results/selections.json) · [Run-level metrics](results/results.csv)

The earlier exploratory pipeline leaked information through cross-store life-week aggregates, overlapping outcome windows and target-week deliveries used to estimate stock. Those provisional scores are excluded. The intermediate corrected package supplies the feature builder and shared model code. Its model outputs are outside this portfolio comparison.

Current-week discount represents a plan that would need to be available at forecast time. Historical data only establishes observed discount values. Start-of-week stock uses deliveries and net sales from prior weeks. Negative net sales targets are clipped to zero. Observed sales can be constrained by stock availability, and the historical test period was reused during project development.

- [Ablations](figures/ablations.png) show the retrained Tweedie models with stock, discount, or potentially unavailable inputs removed.
- [Discount scenarios](figures/response.png) hold other inputs fixed. The curves show model predictions under substituted discounts. They cannot identify causal uplift, latent demand or a profitable pricing policy. [Joint support](figures/support_life.png) shows that 81.29% of the sampled 60% discount substitutions exceeded the validation distance threshold.
- [Robustness](figures/robustness.png) shows three rolling evaluations and 20 training-product resamples with restricted candidate pools and fixed validation observations. The resulting variation is a retrospective sensitivity check.
- [Occurrence](figures/occurrence.png) shows sale/no-sale outcomes using validation-selected thresholds. [Training](figures/training.png) shows optimisation and validation traces.

Additional results include [regression metrics](results/regression_metrics.csv), [rolling folds](results/rolling_folds.json), [joint support](results/joint_support.csv) and [conditional uncertainty](results/conditional_uncertainty.json).
