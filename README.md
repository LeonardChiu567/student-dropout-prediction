# Student Dropout Prediction

Predicting which undergraduate students will drop out, using their socioeconomic and academic data. The project compares logistic regression, random forest and XGBoost, each as an untuned baseline and a tuned model. A missed dropout costs far more than an extra counselor meeting, so each model's decision threshold is set to catch at least 90% of dropouts.

**Result:** the three tuned models are statistically tied (test PR-AUC about 0.96), so tuned logistic regression was chosen for its simplicity and interpretability. On the test set, it caught 251 of 269 dropouts (recall 0.933, precision 0.845) and flagged 297 of 696 students.

## Data

The data is the UCI [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) dataset: 4,424 students from a Portuguese higher-education institution, with 36 features covering demographics, family background, finances and 1st/2nd-semester academic results. It is included in `data/` under its CC BY 4.0 license.

> Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

## Approach

- **Target:** dropout (1) vs. graduate (0). Students still enrolled are excluded, because their outcome is unknown (3,630 students remain).
- **Cleaning:**
  - Removed 152 students with zero enrolled units in both semesters. All of them are in one course, so these look like missing records, not early dropouts. This leaves 3,478 students.
  - Merged categories with fewer than 30 students (about 1% of the training set) into related categories, including two small courses.
- **Features:**
  - Grouped high-cardinality categories (application mode, qualifications, parents' occupation and education) by meaning.
  - Dropped redundant or leaky columns: nationality (duplicates *International*), daytime/evening attendance (determined by course), and the three economic indicators (they act as a year identifier).
  - Replaced raw unit counts with approval and evaluation rates, averaged across both semesters, plus semester-to-semester change features. This brought the maximum VIF down to 10.8.
  - One-hot encoded the categorical features, using the most common category as the reference.
- **Split:** stratified 80/20 train/test split. The 13 continuous features are scaled inside a pipeline.
- **Model selection:**
  - Models were compared on PR-AUC (average precision), with hyperparameters tuned by 5-fold stratified CV on the training set.
  - Each model's threshold is the highest cutoff that reaches at least 90% recall on out-of-fold predictions.
  - The test set was used once, at the end.
- **Sensitivity check:** refit every tuned model without *Tuition fees up to date* and *Debtor*, because the data doesn't say when these were recorded.
- **Uncertainty:** stratified bootstrap of the test set (1,000 resamples) for 95% confidence intervals, plus paired comparisons between models.
- **Interpretation:** statsmodels logistic regression (odds ratios, p-values), permutation importance and SHAP.

## Results (test set: 696 students, 269 dropouts)

| Model | CV PR-AUC | Test PR-AUC | Threshold | Precision | Recall | Caught | Flagged |
|---|---|---|---|---|---|---|---|
| LR baseline | 0.956 | 0.961 | 0.500 | 0.928 | 0.859 | 231 | 249 |
| **LR tuned** | **0.956** | **0.961** | **0.249** | **0.845** | **0.933** | **251** | **297** |
| RF baseline | 0.959 | 0.960 | 0.500 | 0.932 | 0.870 | 234 | 251 |
| RF tuned | 0.960 | 0.961 | 0.294 | 0.854 | 0.914 | 246 | 288 |
| XGB baseline | 0.951 | 0.957 | 0.500 | 0.925 | 0.870 | 234 | 253 |
| XGB tuned | 0.959 | 0.962 | 0.243 | 0.878 | 0.911 | 245 | 279 |

- **Bootstrap:**
  - The tuned models are tied on PR-AUC: every pairwise difference is within ±0.007.
  - Logistic regression's recall is higher than XGBoost's (+0.022, 95% CI [+0.004, +0.045]). That gain comes from flagging more students, not from ranking them better.
- **Sensitivity:** removing fee and debt status lowers CV PR-AUC by less than 0.01 for every model, which is smaller than the fold-to-fold noise. Recall stays above 90%.
- **Top factors:**
  - The average approval rate (share of enrolled units passed) is by far the strongest predictor, and tuition fees being up to date is second. Both lower dropout risk.
  - Being a debtor and enrolling in Basic Education raise the risk; enrolling in Social Service lowers it.
  - Permutation importance and SHAP agree on the top two.
- **Practical takeaway:** contact first the students who pass few of their first-year units or fall behind on fees.

## Limitations

- The data comes from a single institution.
- The data doesn't say when fee and debt status were recorded. The sensitivity check shows that the results don't depend on them.
- Second-semester results partly reflect students who had already left, so the model is somewhat optimistic as a true early-warning system.
- Factors such as mental health aren't in the data.

## Run it

```powershell
pip install -r requirements.txt
cd notebooks
jupyter notebook student_dropout_prediction.ipynb
```
