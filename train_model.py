"""Train logistic regression for loan default and export coefficients for n8n.
Replace the synthetic block with your real CSV (e.g. Kaggle 'Credit Risk Dataset'):
    df = pd.read_csv('loan_data.csv'); X = df[FEATURES]; y = df['default']
"""
import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

FEATURES = ["loan_amount","annual_income","credit_score","interest_rate",
            "dti_ratio","employment_years","late_payments","credit_utilization"]

rng = np.random.default_rng(42); n = 8000
df = pd.DataFrame({
 "loan_amount": rng.uniform(1000, 40000, n),
 "annual_income": rng.lognormal(10.9, 0.45, n),
 "credit_score": np.clip(rng.normal(680, 70, n), 300, 850),
 "interest_rate": rng.uniform(5, 24, n),
 "dti_ratio": np.clip(rng.normal(0.25, 0.12, n), 0.01, 0.9),
 "employment_years": rng.gamma(2.5, 2.5, n),
 "late_payments": rng.poisson(0.8, n),
 "credit_utilization": np.clip(rng.normal(0.4, 0.22, n), 0, 1),
})
z = (-2.0 + 3.0*(df.loan_amount/df.annual_income) - 0.012*(df.credit_score-680)
     + 0.08*(df.interest_rate-12) + 4.0*(df.dti_ratio-0.25) - 0.08*df.employment_years
     + 0.45*df.late_payments + 1.5*(df.credit_utilization-0.4))
df["default"] = (rng.random(n) < 1/(1+np.exp(-z))).astype(int)

X, y = df[FEATURES], df["default"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
p = clf.predict_proba(sc.transform(Xte))[:,1]
print("Accuracy:", round(accuracy_score(yte, p>0.5),3), "AUC:", round(roc_auc_score(yte,p),3))
print(classification_report(yte, p>0.5))
model = {"features":FEATURES,"mean":sc.mean_.round(4).tolist(),"std":sc.scale_.round(4).tolist(),
         "coef":clf.coef_[0].round(4).tolist(),"intercept":round(float(clf.intercept_[0]),4)}
print(json.dumps(model))
json.dump(model, open("model.json","w"))
