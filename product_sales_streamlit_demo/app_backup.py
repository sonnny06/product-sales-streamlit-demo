import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_product_sales_prediction_model.joblib"
META_PATH = BASE_DIR / "models" / "model_metadata.json"
DATA_PATH = BASE_DIR / "data" / "retail_store_inventory.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data_preview():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


def build_model_input(form_data: dict, metadata: dict) -> pd.DataFrame:
    """Create one input row with the same feature columns used during training."""
    selected_date = pd.to_datetime(form_data["Date"])

    row = {
        "Store_ID": form_data["Store_ID"],
        "Product_ID": form_data["Product_ID"],
        "Category": form_data["Category"],
        "Region": form_data["Region"],
        "Inventory_Level": form_data["Inventory_Level"],
        "Units_Ordered": form_data["Units_Ordered"],
        "Price": form_data["Price"],
        "Discount": form_data["Discount"],
        "Weather_Condition": form_data["Weather_Condition"],
        "Holiday_Promotion": form_data["Holiday_Promotion"],
        "Competitor_Pricing": form_data["Competitor_Pricing"],
        "Seasonality": form_data["Seasonality"],
        "Year": selected_date.year,
        "Month": selected_date.month,
        "Day": selected_date.day,
        "DayOfWeek": selected_date.dayofweek,
        "Quarter": selected_date.quarter,
        "Is_Weekend": 1 if selected_date.dayofweek >= 5 else 0,
    }

    row["Discounted_Price"] = row["Price"] * (1 - row["Discount"] / 100)
    row["Price_Difference"] = row["Price"] - row["Competitor_Pricing"]
    row["Inventory_Level_Log"] = np.log1p(row["Inventory_Level"])

    input_df = pd.DataFrame([row])

    # Keep exactly the same columns and order as the model saw during training.
    feature_columns = metadata["feature_columns"]
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = metadata["default_values"].get(col, 0)

    return input_df[feature_columns]


def demand_level(predicted_units: float) -> str:
    if predicted_units >= 220:
        return "High demand"
    if predicted_units >= 110:
        return "Medium demand"
    return "Low demand"


def business_recommendation(predicted_units: float, inventory_level: int) -> str:
    if predicted_units > inventory_level:
        return (
            "Predicted demand is higher than current inventory. "
            "Increase stock level to reduce stockout risk."
        )

    if predicted_units >= 220:
        return (
            "Demand looks strong. Prepare inventory carefully and consider promotion timing."
        )

    if predicted_units < 80:
        return (
            "Demand looks low. Avoid overstocking and review pricing or promotion strategy."
        )

    return (
        "Demand looks stable. Maintain current inventory level and monitor market conditions."
    )


st.set_page_config(
    page_title="Product Sales Prediction",
    page_icon="📈",
    layout="wide",
)

model = load_model()
metadata = load_metadata()
data_preview = load_data_preview()

st.title("📈 Product Sales Prediction Demo")
st.caption("Machine Learning Regression Demo: New Business Data → Model Prediction → Business Decision")

with st.expander("Project information", expanded=False):
    st.write(
        """
        This small web demo uses the trained regression model from the notebook.
        The target variable is **Units Sold**. Users enter new retail business data,
        then the model predicts expected product sales.
        """
    )

    metrics_df = pd.DataFrame(metadata["metrics"])
    st.dataframe(metrics_df, use_container_width=True)

tab_prediction, tab_performance, tab_comparison, tab_explain = st.tabs([
    "Prediction",
    "Model Performance",
    "Algorithm Comparison",
    "Explainability"
])

with tab_prediction:
    left, right = st.columns([1, 1])

    category_options = metadata["category_options"]
    defaults = metadata["default_values"]

    with left:
        st.subheader("1. Input new business data")

        date_value = st.date_input("Date")

        col_a, col_b = st.columns(2)
        with col_a:
            store_id = st.selectbox(
                "Store ID",
                category_options.get("Store_ID", ["S001"]),
                index=0,
            )
            category = st.selectbox(
                "Category",
                category_options.get("Category", ["Electronics"]),
                index=0,
            )
            region = st.selectbox(
                "Region",
                category_options.get("Region", ["North"]),
                index=0,
            )
        with col_b:
            product_id = st.selectbox(
                "Product ID",
                category_options.get("Product_ID", ["P0001"]),
                index=0,
            )
            weather_condition = st.selectbox(
                "Weather Condition",
                category_options.get("Weather_Condition", ["Sunny"]),
                index=0,
            )
            seasonality = st.selectbox(
                "Seasonality",
                category_options.get("Seasonality", ["Summer"]),
                index=0,
            )

        col_c, col_d = st.columns(2)
        with col_c:
            inventory_level = st.number_input(
                "Inventory Level",
                min_value=0,
                max_value=10000,
                value=int(defaults.get("Inventory_Level", 100)),
                step=1,
            )
            units_ordered = st.number_input(
                "Units Ordered",
                min_value=0,
                max_value=10000,
                value=int(defaults.get("Units_Ordered", 100)),
                step=1,
            )
            price = st.number_input(
                "Price",
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults.get("Price", 50.0)),
                step=1.0,
            )

        with col_d:
            discount = st.slider(
                "Discount (%)",
                min_value=0,
                max_value=100,
                value=int(defaults.get("Discount", 10)),
            )
            competitor_pricing = st.number_input(
                "Competitor Pricing",
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults.get("Competitor_Pricing", 50.0)),
                step=1.0,
            )
            holiday_promotion = st.selectbox(
                "Holiday / Promotion",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                index=int(defaults.get("Holiday_Promotion", 0)),
            )

        predict_button = st.button("Predict Units Sold", type="primary", use_container_width=True)

    form_data = {
        "Date": date_value,
        "Store_ID": store_id,
        "Product_ID": product_id,
        "Category": category,
        "Region": region,
        "Inventory_Level": inventory_level,
        "Units_Ordered": units_ordered,
        "Price": price,
        "Discount": discount,
        "Weather_Condition": weather_condition,
        "Holiday_Promotion": holiday_promotion,
        "Competitor_Pricing": competitor_pricing,
        "Seasonality": seasonality,
    }

    with right:
        st.subheader("2. Prediction output")

        input_df = build_model_input(form_data, metadata)

        if predict_button:
            prediction = float(model.predict(input_df)[0])
            prediction = max(0, prediction)

            st.metric("Predicted Units Sold", f"{prediction:,.0f} units")
            st.info(f"Demand Level: **{demand_level(prediction)}**")
            st.success(business_recommendation(prediction, inventory_level))

            st.write("Model input after feature engineering:")
            st.dataframe(input_df, use_container_width=True)
        else:
            st.write("Enter business data, then click **Predict Units Sold**.")



with tab_performance:
    st.header("Model Performance")

    best_metrics = metadata.get("best_model_metrics", {})

    if best_metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", round(best_metrics.get("MAE", 0), 2))
        col2.metric("RMSE", round(best_metrics.get("RMSE", 0), 2))
        col3.metric("R² Score", round(best_metrics.get("R2_Score", 0), 4))

        st.info(
            """
            MAE and RMSE measure prediction error. Lower values are better.
            R² Score measures how well the model explains the variation in Units Sold.
            """
        )
    else:
        st.warning("Best model metrics are not available. Please run train_model.py again.")

    actual_vs_predicted = metadata.get("actual_vs_predicted", [])

    if actual_vs_predicted:
        actual_pred_df = pd.DataFrame(actual_vs_predicted)

        st.subheader("Actual vs Predicted Units Sold")

        fig_actual_pred = px.scatter(
            actual_pred_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Units Sold",
            labels={
                "Actual": "Actual Units Sold",
                "Predicted": "Predicted Units Sold"
            }
        )

        st.plotly_chart(fig_actual_pred, use_container_width=True)
        st.dataframe(actual_pred_df.head(20), use_container_width=True)
    else:
        st.warning("Actual vs Predicted data is not available. Please run train_model.py again.")


with tab_comparison:
    st.header("Algorithm Comparison")

    metrics_df = pd.DataFrame(metadata["metrics"])

    st.subheader("Model Evaluation Table")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Comparison by R² Score")

    fig_r2 = px.bar(
        metrics_df,
        x="Model",
        y="R2_Score",
        text="R2_Score",
        title="Algorithm Comparison by R² Score"
    )

    st.plotly_chart(fig_r2, use_container_width=True)

    st.subheader("Comparison by RMSE")

    fig_rmse = px.bar(
        metrics_df,
        x="Model",
        y="RMSE",
        text="RMSE",
        title="Algorithm Comparison by RMSE"
    )

    st.plotly_chart(fig_rmse, use_container_width=True)

    st.success(
        f"""
        The selected final model is **{metadata['best_model_name']}**.
        It is selected based on lower RMSE and strong overall evaluation performance.
        """
    )


with tab_explain:
    st.header("Model Explainability")

    st.write(
        """
        This section shows which input features have the strongest influence
        on the prediction of Units Sold.
        """
    )

    feature_importance = metadata.get("feature_importance", [])

    if feature_importance:
        feature_df = pd.DataFrame(feature_importance)

        st.subheader("Top Feature Importance")

        fig_feature = px.bar(
            feature_df.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Feature Importance"
        )

        st.plotly_chart(fig_feature, use_container_width=True)
        st.dataframe(feature_df, use_container_width=True)

        top_feature = feature_df.iloc[0]["Feature"]

        st.info(
            f"""
            The most influential feature is **{top_feature}**.
            This means the model relies heavily on this variable when predicting Units Sold.
            """
        )
    else:
        st.warning(
            """
            Feature importance is not available. Please run train_model.py again,
            or check whether the selected model supports feature importance.
            """
        )

st.divider()

bottom_left, bottom_right = st.columns([1, 1])

with bottom_left:
    st.subheader("Inference Flow")
    st.write("**New Business Data → Preprocessing → Trained Regression Model → Predicted Units Sold → Business Decision**")

with bottom_right:
    st.subheader("Dataset Preview")
    if not data_preview.empty:
        st.dataframe(data_preview.head(10), use_container_width=True)
    else:
        st.write("Dataset file was not found.")
