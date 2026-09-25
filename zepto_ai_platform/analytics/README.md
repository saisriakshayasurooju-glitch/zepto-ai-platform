# Module 2 — Analytics

Run:

```bash
python analytics/analysis.py
```

The script saves `titanic.csv` immediately after loading the dataset and then performs EDA, missing-value profiling, univariate and multivariate visualization, correlation analysis, classification, imbalance experiments, Random Forest grid search, fare regression, residual analysis and pipeline persistence.

Generated figures are saved under `analytics/figures/`, and the fitted classification pipeline is saved as `analytics/best_pipeline.joblib`.
