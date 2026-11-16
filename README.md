# Loan Default Prediction (n8n + Streamlit)

Predicts whether a borrower is likely to default using financial and repayment features.

- `n8n/loan_default_workflow.json` – n8n workflow (Webhook -> logistic model -> HTML/JSON response)
- `app.py` – Streamlit UI (sliders -> predicted default class)
- `train_model.py` – trains the model and writes `model.json`
- `model.json` – exported coefficients, mean and std used for scoring

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy
Push to GitHub, then create an app at https://share.streamlit.io pointing to `app.py`.

## Demo
Change one feature (e.g. credit score 690 -> 520) and observe the class flip from 0 to 1.
