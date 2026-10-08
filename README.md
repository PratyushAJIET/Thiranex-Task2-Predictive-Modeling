# Thiranex Task 2 — Predictive Modeling Using Machine Learning

**Student:** Pratyush  
**Task:** Predictive Modeling Using Machine Learning  
**Project:** Employee Attrition Classification with Logistic Regression and Random Forest

## Dataset
`employee_attrition_data.csv` contains **650 fictional employee records** created programmatically for educational use. **No real employee or company data is included.** Labels are simulated and do not reflect real-world employment decisions. Use only if the internship permits a self-created dataset.

## Aim
Predict the binary label `Attrition` (Yes/No). Compare two supervised classifiers and explain performance with confusion matrices, ROC curves and AUC, following the concepts covered in the supplied Thiranex tutorial. The tutorial itself uses microscopy-image classification; this project is a simpler independent tabular classification adaptation, **not** an exact reconstruction of the video.

## Workflow
1. Import and inspect the data; remove the identifier from features.
2. Split into stratified 80% training and 20% testing.
3. Encode department and overtime, scale numeric columns using train-only preprocessing pipelines.
4. Fit Logistic Regression and Random Forest.
5. Report accuracy, precision, recall, F1 and ROC AUC on test data.
6. Plot confusion matrices and ROC curves, then explore decision thresholds 0.30, 0.50 and 0.70.

## Run in Google Colab
1. Upload `predictive_model.ipynb` to Colab.
2. Open the left **Files** panel and upload `employee_attrition_data.csv`.
3. Click **Runtime → Run all**.
4. Confirm graphs appear and download the executed notebook, charts, and generated CSVs.

Or run locally with `pip install -r requirements.txt`, then open the notebook in Jupyter.

## Outputs
- `predictive_model.ipynb` — complete executed Python notebook.
- `employee_attrition_data.csv` — synthetic input data.
- `model_metrics.csv` — model comparison metrics.
- `threshold_comparison.csv` — threshold effects on TP/FP/TN/FN and precision/recall.
- `charts/confusion_matrices.png` — per-model confusion matrices.
- `charts/roc_curves.png` — ROC curves / AUC.
- `charts/threshold_tradeoff.png` — precision/recall vs threshold.

## Findings
See `model_metrics.csv` and the notebook for actual computed results. Higher ROC AUC means better ranking of positive vs negative cases on this fictional holdout dataset. The threshold trade-off demonstrates that lowering the decision threshold can improve recall but increase false alarms. This is a learning demonstration, **not a validated real-world HR decision system**.

## Reference
Thiranex Task 2 tutorial: https://youtu.be/Joh3LOaG8Q0 (confusion matrices, ROC and AUC).
