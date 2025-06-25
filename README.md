# AccountValue-api


This project provides a **Flask-based REST API** for predicting the account value of a business customer based on their profile and insurance product information. The API uses a machine learning model trained on historical insurance data.

## Features

- **/predict** endpoint: Accepts business and product details as JSON and returns a predicted account value.
- **Preprocessing**: Matches the feature engineering and encoding logic used during model training.
- **Model**: XGBoost regressor trained on engineered features.
- **Artifacts**: Model, encoders, and metadata are loaded at startup.

## Usage

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Run the API

```sh
python app.py
```

The API will be available at `https://account-value-api.onrender.com/predict`.

### 3. Example API Call

```sh
curl -X POST https://account-value-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "account_uuid": "b02a677c-25e7-4dbf-b52d-13063fc0dfa3",
    "product": ["Business_Owners_Policy_BOP"],
    "state": "KS",
    "industry": "Coin-Operated Laundries and Drycleaners",
    "subindustry": "Coin-Operated Laundries and Drycleaners",
    "business_structure": "Corporation",
    "year_established": 2018,
    "annual_revenue": 100000,
    "total_payroll": 22000,
    "num_employees": 3,
    "avg_premium": 507
  }'
```

### 4. Project Structure

- `app.py` — Flask API application
- `preprocessing.py` — Feature engineering and preprocessing logic
- `train_model.py` — Model training and artifact export
- `requirements.txt` — Python dependencies
- `model.pkl`, `ordinal_encoders.pkl`, `metadata.pkl` — Model and preprocessing artifacts

## Development

- Update `train_model.py` and retrain if you want to improve the model.
- Make sure to keep preprocessing logic in sync between training and inference.
- Use the provided GitHub Actions workflow for CI checks.
