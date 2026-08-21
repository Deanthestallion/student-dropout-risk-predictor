"""Build data/sample_class.csv: a realistic 30-student register for the batch demo.

Rows are real students drawn from the held-out test split, so the batch tab
demonstrates on data the model never trained on.
"""
import joblib, pandas as pd
from sklearn.model_selection import train_test_split

art = joblib.load("models/dropout_risk_model.joblib")
RAW, cfg = art["feature_names_raw"], art["config"]

df = pd.read_csv("data/student-por.csv", sep=";")
y = (df["G3"] < cfg["pass_mark"]).astype(int)
X = df[RAW]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["random_state"])

# 30 students keeping roughly the real at-risk prevalence
at_risk = X_test[y_test == 1].head(5)
rest = X_test[y_test == 0].head(25)
sample = pd.concat([at_risk, rest]).sample(frac=1, random_state=7).reset_index(drop=True)
sample.insert(0, "student_id", [f"STU-{i:03d}" for i in range(1, len(sample) + 1)])

sample.to_csv("data/sample_class.csv", index=False)
print(f"Wrote data/sample_class.csv: {sample.shape[0]} students x {sample.shape[1]} columns")
print(f"Columns: student_id + {len(RAW)} model features")
print(sample[["student_id", "G1", "absences", "failures", "studytime"]].head().to_string(index=False))
