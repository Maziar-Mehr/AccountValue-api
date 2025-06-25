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
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "year_established": 2005,
  "total_payroll": 500000,
  "num_employees": 25,
  "annual_revenue": 1200000,
  "state": "California",
  "industry": "Retail",
  "subindustry": "Clothing Stores",
  "business_structure": "LLC",
  "product_list": ["General Liability", "Workers Compensation"],
  "avg_premium": 3500
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
