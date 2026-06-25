"""
Training script for the Streamlit demo.

Important note:
- The original notebook `notebook/rut_gon.ipynb` is kept 100% unchanged.
- This script follows the same ML workflow from `rut_gon.ipynb` so the web app can retrain and reuse the model.
- Only notebook-only display/plotting lines are removed from this script.
- The notebook's final saving cell refers to `final_model`; here, `final_model` is explicitly assigned
  from the best result in `final_results_df` so the model can be saved reliably.
"""

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "retail_store_inventory.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Cell 3: Clean column names
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    target_col = "Units_Sold"
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    # Cell 4: Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=[target_col])
    df = df[df[target_col] >= 0]

    # Cell 5: IQR outlier handling for target
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]

    # Cell 6: Date feature engineering
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Quarter"] = df["Date"].dt.quarter
        df["Is_Weekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
        df = df.drop(columns=["Date"])

    # Cell 6 + Cell 7: Feature engineering
    # The notebook repeats these feature engineering lines twice.
    # Repeating them produces the same final values, so one execution is enough here.
    if "Price" in df.columns and "Discount" in df.columns:
        df["Discounted_Price"] = df["Price"] * (1 - df["Discount"] / 100)

    if "Price" in df.columns and "Competitor_Pricing" in df.columns:
        df["Price_Difference"] = df["Price"] - df["Competitor_Pricing"]

    if "Inventory_Level" in df.columns:
        df["Inventory_Level_Log"] = np.log1p(df["Inventory_Level"])

    # Cell 8: Drop leakage column
    leakage_cols = []
    for col in ["Demand_Forecast"]:
        if col in df.columns:
            leakage_cols.append(col)
    df = df.drop(columns=leakage_cols, errors="ignore")

    return df


def evaluate_model(model_name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "Model": model_name,
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2_Score": float(r2),
    }


def main():
    # Cell 1: Load dataset
    data_path = DATA_PATH
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)

    target_col = "Units_Sold"
    df = clean_and_prepare_data(df)
    print("Dataset shape after cleaning and feature engineering:", df.shape)

    # Cell 9: Split X/y and feature types
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Cell 10: Processing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        sparse_threshold=0
    )

    # Cell 11: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # Cell 12: Models, kept the same as rut_gon.ipynb
    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42,
            max_depth=10
        ),

        "Random Forest": RandomForestRegressor(
            random_state=42,
            n_estimators=100,
            max_depth=15,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
    }

    # Cell 13: Train and compare models
    results = []
    trained_models = {}

    for model_name, model in models.items():
        print("=" * 60)
        print("Training model:", model_name)

        pipeline = Pipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        result = evaluate_model(model_name, pipeline, X_test, y_test)

        results.append(result)
        trained_models[model_name] = pipeline

        print("MAE:", result["MAE"])
        print("MSE:", result["MSE"])
        print("RMSE:", result["RMSE"])
        print("R2 Score:", result["R2_Score"])

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="RMSE", ascending=True)

    # Cell 14: GridSearchCV for Random Forest, kept the same as rut_gon.ipynb
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "model__n_estimators": [100, 150],
        "model__max_depth": [10, 15],
        "model__min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    print("=" * 60)
    print("Training model: Tuned Random Forest")
    grid_search.fit(X_train, y_train)

    print("Best parameters:")
    print(grid_search.best_params_)

    print("Best CV RMSE:")
    print(-grid_search.best_score_)

    tuned_model = grid_search.best_estimator_
    tuned_result = evaluate_model("Tuned Random Forest", tuned_model, X_test, y_test)

    final_results_df = pd.concat([results_df, pd.DataFrame([tuned_result])], ignore_index=True)
    final_results_df = final_results_df.sort_values(by="RMSE", ascending=True)

    # Notebook compatibility fix:
    # rut_gon.ipynb saves `final_model`, but the variable is not explicitly assigned in the notebook.
    # Here, final_model is assigned from the best row in final_results_df.
    best_model_name = final_results_df.iloc[0]["Model"]
    if best_model_name == "Tuned Random Forest":
        final_model = tuned_model
    else:
        final_model = trained_models[best_model_name]

    print("=" * 60)
    print("Final results:")
    print(final_results_df)
    print("Best model:", best_model_name)

    # Cell 15: Save model
    model_filename = MODEL_DIR / "best_product_sales_prediction_model.joblib"
    joblib.dump(final_model, model_filename)
    print("Model saved successfully:", model_filename)

    loaded_model = joblib.load(model_filename)
    loaded_predictions = loaded_model.predict(X_test.head(5))
    print(pd.DataFrame({
        "Actual": y_test.head(5).values,
        "Predicted_by_Loaded_Model": loaded_predictions
    }))

    # Metadata for Streamlit app dropdowns and input reconstruction
    metadata = {
        "target_col": target_col,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_model_name": str(best_model_name),
        "metrics": final_results_df.to_dict(orient="records"),
        "best_params": grid_search.best_params_,
        "category_options": {},
        "default_values": {},
    }

    for col in categorical_features:
        metadata["category_options"][col] = sorted(
            df[col].dropna().astype(str).unique().tolist()
        )[:300]

    for col in X.columns:
        if col in numeric_features:
            metadata["default_values"][col] = float(X[col].median())
        else:
            mode_value = X[col].mode(dropna=True)
            metadata["default_values"][col] = str(mode_value.iloc[0]) if len(mode_value) else ""

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Metadata saved successfully:", MODEL_DIR / "model_metadata.json")


if __name__ == "__main__":
    main()
