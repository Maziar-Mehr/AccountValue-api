# src/preprocessing.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OrdinalEncoder

# --- Helper for state abbreviation (copied from notebook for 100% match) ---
us_state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'New York': 'NY', 'Oregon': 'OR', 'Florida': 'FL',
    'Washington DC': 'DC', 'Pennsylvania': 'PA', 'PA - Pennsylvania': 'PA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN',
    'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

def load_model_and_preprocessing_objects(model_path: str, preprocessing_path: str):
    """Loads the trained model and preprocessing objects."""
    try:
        model = joblib.load(model_path)
        preprocessing_objects = joblib.load(preprocessing_path)
        print("Model and preprocessing objects loaded successfully.")
        return model, preprocessing_objects
    except FileNotFoundError:
        raise FileNotFoundError(f"One or both files not found: {model_path}, {preprocessing_path}. Ensure they are in the correct directory relative to api.py.")
    except Exception as e:
        raise RuntimeError(f"Error loading model or preprocessing objects: {e}")

def preprocess_new_data(input_df: pd.DataFrame, preprocessing_objects: dict) -> pd.DataFrame:
    """100% notebook-matching preprocessing for new input data."""

    df = input_df.copy()

    # --- Extract preprocessing components ---
    ordinal_encoders = preprocessing_objects['ordinal_encoders']
    product_rank_map = preprocessing_objects['product_rank_map']
    median_product_rank = preprocessing_objects['median_product_rank']
    sub_industry_rank_map = preprocessing_objects['sub_industry_rank_map']
    median_sub_industry_rank = preprocessing_objects['median_sub_industry_rank']
    state_total_value_map = preprocessing_objects['state_total_value_map']
    median_total_state_account_value = preprocessing_objects['median_total_state_account_value']
    X_train_columns = preprocessing_objects['X_train_columns']
    # For 100% match: get medians for numeric columns if present
    medians = preprocessing_objects.get('medians', {})

    # --- Ensure all expected columns are present ---
    expected_raw_input_cols = [
        'year_established', 'total_payroll', 'num_employees', 'annual_revenue',
        'state', 'industry', 'subindustry', 'business_structure', 'product_list', 'avg_premium'
    ]
    for col in expected_raw_input_cols:
        if col not in df.columns:
            if col == 'product_list':
                df[col] = [[] for _ in range(len(df))]
            else:
                df[col] = np.nan

    # --- State abbreviation mapping (notebook logic) ---
    df['state'] = df['state'].map(us_state_abbrev).fillna(df['state'])

    # --- Product Premium Rank ---
    df['product_ranks_list'] = df['product_list'].apply(
        lambda products: [product_rank_map.get(p, median_product_rank) for p in (products if isinstance(products, list) else []) if p != 'Unknown']
    )
    df['avg_product_rank'] = df['product_ranks_list'].apply(
        lambda x: np.mean(x) if x else median_product_rank
    )
    df.drop(columns=['product_ranks_list'], inplace=True)

    # --- Subindustry Premium Rank ---
    df['subindustry_rank'] = df['subindustry'].map(sub_industry_rank_map)
    df['subindustry_rank'] = df['subindustry_rank'].fillna(median_sub_industry_rank)

    # --- Total State Account Value ---
    df['total_state_account_value'] = df['state'].map(state_total_value_map).fillna(median_total_state_account_value)

    # --- log1p transformation for avg_premium and annual_revenue (notebook order) ---
    df['avg_premium_log'] = np.log1p(df['avg_premium'].fillna(0.0))
    df['annual_revenue_log'] = np.log1p(df['annual_revenue'].fillna(0.0))
    df = df.drop(columns=['avg_premium', 'annual_revenue'])

    # --- One-hot encode exploded product types (for product flags, notebook logic) ---
    temp_uuid_col = 'temp_account_uuid'
    df[temp_uuid_col] = range(len(df))
    df_exploded = df[[temp_uuid_col, 'product_list']].explode('product_list')
    product_dummies = pd.get_dummies(df_exploded['product_list'])
    product_dummies[temp_uuid_col] = df_exploded[temp_uuid_col]
    product_flags = product_dummies.groupby(temp_uuid_col).max().reset_index()
    # Align product flag columns to match training columns (notebook logic)
    train_product_flag_cols = [col for col in X_train_columns if col not in [
        'year_established', 'total_payroll', 'num_employees', 'state', 'industry', 'business_structure',
        'subindustry_rank', 'avg_product_rank', 'total_state_account_value', 'avg_premium_log', 'annual_revenue_log'
    ]]
    aligned_product_flags = pd.DataFrame(0, index=product_flags.index, columns=train_product_flag_cols)
    for col in product_flags.columns:
        if col in aligned_product_flags.columns:
            aligned_product_flags[col] = product_flags[col]
    aligned_product_flags[temp_uuid_col] = product_flags[temp_uuid_col]
    df = df.drop(columns=['product_list'])
    df = df.merge(aligned_product_flags, on=temp_uuid_col, how='left')
    df = df.drop(columns=[temp_uuid_col])

    # --- Fill missing numeric columns with training medians (notebook logic) ---
    numeric_cols_to_fill = [
        'year_established', 'total_payroll', 'num_employees',
        'avg_product_rank', 'subindustry_rank', 'total_state_account_value',
        'avg_premium_log', 'annual_revenue_log'
    ]
    for col in numeric_cols_to_fill:
        if col in df.columns:
            median_val = medians.get(col, 0.0)
            df[col] = df[col].fillna(median_val)

    # --- Fill missing categoricals with 'Unknown' (notebook logic) ---
    categorical_cols_to_fill_and_encode = ['state', 'industry', 'business_structure']
    for col in categorical_cols_to_fill_and_encode:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # --- Ordinal encoding (notebook logic) ---
    for col in categorical_cols_to_fill_and_encode:
        if col in df.columns:
            df[col] = ordinal_encoders[col].transform(df[[col]])

    # --- Remove subindustry (notebook logic) ---
    if 'subindustry' in df.columns:
        df = df.drop(columns=['subindustry'])

    # --- Final column alignment (notebook logic) ---
    final_features_df = df.reindex(columns=X_train_columns, fill_value=0)

    # --- Ensure all numeric types for XGBoost ---
    for col in final_features_df.columns:
        if final_features_df[col].dtype == 'bool':
            final_features_df[col] = final_features_df[col].astype(int)
        elif pd.api.types.is_numeric_dtype(final_features_df[col]):
            final_features_df[col] = final_features_df[col].astype(float)
        else:
            final_features_df[col] = pd.to_numeric(final_features_df[col], errors='coerce').fillna(0.0)

    return final_features_df