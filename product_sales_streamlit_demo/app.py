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
    """Tạo một dòng dữ liệu đầu vào có đúng cấu trúc cột như lúc huấn luyện model."""
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

    # Giữ đúng thứ tự cột như lúc train để tránh lỗi khi predict.
    feature_columns = metadata["feature_columns"]
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = metadata["default_values"].get(col, 0)

    return input_df[feature_columns]


def demand_level(predicted_units: float) -> str:
    if predicted_units >= 220:
        return "Nhu cầu cao"
    if predicted_units >= 110:
        return "Nhu cầu trung bình"
    return "Nhu cầu thấp"


def business_recommendation(predicted_units: float, inventory_level: int) -> str:
    if predicted_units > inventory_level:
        return (
            "Nhu cầu dự đoán cao hơn lượng tồn kho hiện tại. "
            "Doanh nghiệp nên tăng mức tồn kho để giảm rủi ro thiếu hàng."
        )

    if predicted_units >= 220:
        return (
            "Nhu cầu dự đoán đang khá mạnh. Doanh nghiệp nên chuẩn bị tồn kho cẩn thận "
            "và cân nhắc thời điểm khuyến mãi phù hợp."
        )

    if predicted_units < 80:
        return (
            "Nhu cầu dự đoán đang thấp. Doanh nghiệp nên tránh nhập hàng quá nhiều "
            "và xem xét lại giá bán hoặc chiến lược khuyến mãi."
        )

    return (
        "Nhu cầu dự đoán tương đối ổn định. Doanh nghiệp có thể duy trì mức tồn kho hiện tại "
        "và tiếp tục theo dõi điều kiện thị trường."
    )


def metric_conclusion(metrics: dict) -> str:
    r2 = metrics.get("R2_Score", metrics.get("R2", 0))
    mae = metrics.get("MAE", 0)
    rmse = metrics.get("RMSE", 0)

    if r2 >= 0.85:
        quality = "mô hình có khả năng giải thích dữ liệu khá tốt"
    elif r2 >= 0.65:
        quality = "mô hình có mức độ phù hợp tương đối ổn"
    elif r2 >= 0.4:
        quality = "mô hình chỉ giải thích được dữ liệu ở mức trung bình"
    else:
        quality = "mô hình còn yếu và cần được cải thiện thêm"

    return (
        f"Với R² ≈ {r2:.3f}, {quality}. "
        f"MAE ≈ {mae:.2f} cho biết trung bình mô hình dự đoán lệch khoảng {mae:.2f} đơn vị bán. "
        f"RMSE ≈ {rmse:.2f} thường lớn hơn MAE vì chỉ số này phạt mạnh các lỗi dự đoán lớn."
    )


def algorithm_conclusion(metrics_df: pd.DataFrame, best_model_name: str) -> str:
    best_rmse_row = metrics_df.sort_values(by="RMSE", ascending=True).iloc[0]
    best_r2_row = metrics_df.sort_values(by="R2_Score", ascending=False).iloc[0]

    return (
        f"Dựa trên bảng so sánh, mô hình được chọn là **{best_model_name}**. "
        f"Mô hình có RMSE thấp nhất là **{best_rmse_row['Model']}** "
        f"với RMSE ≈ {best_rmse_row['RMSE']:.2f}. "
        f"Mô hình có R² cao nhất là **{best_r2_row['Model']}** "
        f"với R² ≈ {best_r2_row['R2_Score']:.3f}. "
        "Nhìn chung, mô hình tốt nên có **RMSE thấp** và **R² cao**."
    )


def feature_conclusion(feature_df: pd.DataFrame) -> str:
    top = feature_df.iloc[0]
    top_feature = str(top["Feature"])
    top_importance = float(top["Importance"])

    return (
        f"Biến có ảnh hưởng mạnh nhất là **{top_feature}** "
        f"với importance ≈ {top_importance:.4f}. "
        "Điều này cho thấy khi dự đoán số lượng bán ra, mô hình phụ thuộc nhiều vào biến này hơn các biến còn lại. "
        "Tuy nhiên, feature importance chỉ cho biết mức độ ảnh hưởng tương đối, không khẳng định quan hệ nhân quả tuyệt đối."
    )


def actual_pred_conclusion(actual_pred_df: pd.DataFrame) -> str:
    if actual_pred_df.empty:
        return "Chưa có dữ liệu Actual vs Predicted để kết luận."

    error = (actual_pred_df["Actual"] - actual_pred_df["Predicted"]).abs()
    avg_error = error.mean()

    return (
        f"Biểu đồ Actual vs Predicted dùng để kiểm tra độ lệch giữa giá trị thực tế và giá trị dự đoán. "
        f"Trong mẫu hiển thị, sai lệch tuyệt đối trung bình khoảng {avg_error:.2f} đơn vị. "
        "Nếu các điểm nằm càng gần đường xu hướng chéo, mô hình dự đoán càng sát thực tế."
    )


st.set_page_config(
    page_title="Dashboard Dự Báo Doanh Số Sản Phẩm",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #102a43 0%, #1f4e5f 55%, #243b53 100%);
        color: white;
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.92;
        margin-bottom: 0;
    }

    .section-card {
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(255,255,255,0.04);
        margin-bottom: 1rem;
    }

    .small-note {
        font-size: 0.92rem;
        opacity: 0.86;
        line-height: 1.5;
    }

    .step-badge {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: rgba(255, 120, 90, 0.18);
        color: #ffb199;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stMetric"] {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(255,255,255,0.04);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.55rem 1rem;
        background: rgba(128,128,128,0.12);
    }

    .stTabs [aria-selected="true"] {
        background: #ff6b57 !important;
        color: white !important;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()
metadata = load_metadata()
data_preview = load_data_preview()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📊 Dashboard Dự Báo Doanh Số Sản Phẩm</div>
        <p class="hero-subtitle">
            Demo Machine Learning Regression: Dữ liệu kinh doanh mới → Tiền xử lý → Dự đoán → Gợi ý quyết định kinh doanh
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Tóm tắt dự án")
    st.write("**Biến mục tiêu:** Units Sold")
    st.write(f"**Mô hình tốt nhất:** {metadata.get('best_model_name', 'N/A')}")
    st.write("**Mục tiêu:** dự báo số lượng bán ra để hỗ trợ quản lý tồn kho và marketing.")
    st.divider()
    st.caption("Dùng các tab để xem dự đoán, độ tin cậy mô hình, so sánh thuật toán và giải thích kết quả.")

with st.expander("Thông tin dự án và bảng đánh giá mô hình", expanded=False):
    st.write(
        """
        Ứng dụng này sử dụng mô hình hồi quy để dự đoán **Units Sold** dựa trên dữ liệu bán lẻ.
        Ngoài kết quả dự đoán, dashboard còn hiển thị các chỉ số và biểu đồ để giải thích độ tin cậy của mô hình.
        """
    )

    metrics_df = pd.DataFrame(metadata["metrics"])
    st.dataframe(metrics_df, use_container_width=True)

tab_prediction, tab_performance, tab_comparison, tab_explain = st.tabs([
    "Dự đoán",
    "Độ tin cậy mô hình",
    "So sánh thuật toán",
    "Giải thích mô hình"
])


with tab_prediction:
    st.markdown(
        """
        <div class="section-card">
            <span class="step-badge">Bước 1</span>
            <h3 style="margin-top:0;">Dự đoán nhu cầu bán hàng</h3>
            <p class="small-note">
                Nhập thông tin kinh doanh mới. Pipeline đã huấn luyện sẽ tiền xử lý dữ liệu,
                tạo đặc trưng bổ sung và dự đoán số lượng sản phẩm có thể bán ra.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    category_options = metadata["category_options"]
    defaults = metadata["default_values"]

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Thông tin đầu vào")

        date_value = st.date_input("Ngày")

        col_a, col_b = st.columns(2)
        with col_a:
            store_id = st.selectbox(
                "Mã cửa hàng",
                category_options.get("Store_ID", ["S001"]),
                index=0,
            )
            category = st.selectbox(
                "Danh mục sản phẩm",
                category_options.get("Category", ["Electronics"]),
                index=0,
            )
            region = st.selectbox(
                "Khu vực",
                category_options.get("Region", ["North"]),
                index=0,
            )
        with col_b:
            product_id = st.selectbox(
                "Mã sản phẩm",
                category_options.get("Product_ID", ["P0001"]),
                index=0,
            )
            weather_condition = st.selectbox(
                "Điều kiện thời tiết",
                category_options.get("Weather_Condition", ["Sunny"]),
                index=0,
            )
            seasonality = st.selectbox(
                "Mùa vụ",
                category_options.get("Seasonality", ["Summer"]),
                index=0,
            )

        col_c, col_d = st.columns(2)
        with col_c:
            inventory_level = st.number_input(
                "Mức tồn kho",
                min_value=0,
                max_value=10000,
                value=int(defaults.get("Inventory_Level", 100)),
                step=1,
            )
            units_ordered = st.number_input(
                "Số lượng đặt hàng",
                min_value=0,
                max_value=10000,
                value=int(defaults.get("Units_Ordered", 100)),
                step=1,
            )
            price = st.number_input(
                "Giá bán",
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults.get("Price", 50.0)),
                step=1.0,
            )

        with col_d:
            discount = st.slider(
                "Giảm giá (%)",
                min_value=0,
                max_value=100,
                value=int(defaults.get("Discount", 10)),
            )
            competitor_pricing = st.number_input(
                "Giá đối thủ",
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults.get("Competitor_Pricing", 50.0)),
                step=1.0,
            )
            holiday_promotion = st.selectbox(
                "Ngày lễ / Khuyến mãi",
                options=[0, 1],
                format_func=lambda x: "Có" if x == 1 else "Không",
                index=int(defaults.get("Holiday_Promotion", 0)),
            )

        predict_button = st.button("Dự đoán Units Sold", type="primary", use_container_width=True)

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
        st.subheader("Kết quả dự đoán")

        input_df = build_model_input(form_data, metadata)

        if predict_button:
            prediction = float(model.predict(input_df)[0])
            prediction = max(0, prediction)

            st.metric("Predicted Units Sold", f"{prediction:,.0f} units")
            st.info(f"Mức nhu cầu: **{demand_level(prediction)}**")
            st.success(business_recommendation(prediction, inventory_level))

            st.write("Dữ liệu sau khi feature engineering:")
            st.dataframe(input_df, use_container_width=True)
        else:
            st.write("Nhập dữ liệu kinh doanh, sau đó bấm **Dự đoán Units Sold**.")

    st.divider()

    bottom_left, bottom_right = st.columns([1, 1])

    with bottom_left:
        st.subheader("Luồng suy luận của mô hình")
        st.write("**Dữ liệu kinh doanh mới → Tiền xử lý → Mô hình hồi quy đã huấn luyện → Predicted Units Sold → Quyết định kinh doanh**")
        st.caption(
            "Ý nghĩa: dữ liệu đầu vào không được đưa thẳng vào mô hình. Trước đó, dữ liệu được xử lý thiếu, mã hóa biến phân loại, chuẩn hóa biến số và tạo các đặc trưng như giá sau giảm, chênh lệch giá với đối thủ, log tồn kho."
        )

    with bottom_right:
        st.subheader("Xem trước dữ liệu gốc")
        if not data_preview.empty:
            st.dataframe(data_preview.head(10), use_container_width=True)
        else:
            st.write("Không tìm thấy file dữ liệu.")


with tab_performance:
    st.markdown(
        """
        <div class="section-card">
            <span class="step-badge">Bước 2</span>
            <h3 style="margin-top:0;">Đánh giá độ tin cậy của mô hình</h3>
            <p class="small-note">
                Phần này trả lời câu hỏi: mô hình dự đoán có sát thực tế không và sai số khoảng bao nhiêu?
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Chỉ số đánh giá mô hình")

    best_metrics = metadata.get("best_model_metrics", {})

    if best_metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", round(best_metrics.get("MAE", 0), 2))
        col2.metric("RMSE", round(best_metrics.get("RMSE", 0), 2))
        col3.metric("R² Score", round(best_metrics.get("R2_Score", best_metrics.get("R2", 0)), 4))

        st.info(
            """
            **Ý nghĩa chỉ số:**
            - **MAE**: sai số tuyệt đối trung bình. MAE càng thấp thì mô hình dự đoán càng sát.
            - **RMSE**: căn bậc hai của sai số bình phương trung bình. RMSE phạt mạnh các lỗi dự đoán lớn.
            - **R² Score**: cho biết mô hình giải thích được bao nhiêu phần biến động của Units Sold. R² càng cao càng tốt.
            """
        )

        st.success("**Kết luận từ chỉ số:** " + metric_conclusion(best_metrics))
    else:
        st.warning("Chưa có best_model_metrics. Hãy chạy lại train_model.py để tạo metadata mới.")

    actual_vs_predicted = metadata.get("actual_vs_predicted", [])

    if actual_vs_predicted:
        actual_pred_df = pd.DataFrame(actual_vs_predicted)

        st.subheader("Biểu đồ Actual vs Predicted")

        fig_actual_pred = px.scatter(
            actual_pred_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Units Sold",
            labels={
                "Actual": "Units Sold thực tế",
                "Predicted": "Units Sold dự đoán"
            },
            trendline="ols"
        )

        st.plotly_chart(fig_actual_pred, use_container_width=True)

        st.info(
            """
            **Cách đọc biểu đồ:**
            - Mỗi điểm là một quan sát trong tập test.
            - Trục X là số lượng bán thực tế.
            - Trục Y là số lượng mô hình dự đoán.
            - Nếu điểm nằm càng gần đường xu hướng chéo, mô hình dự đoán càng sát thực tế.
            - Điểm cách xa đường xu hướng là các trường hợp mô hình dự đoán lệch nhiều.
            """
        )

        st.success("**Kết luận từ biểu đồ:** " + actual_pred_conclusion(actual_pred_df))
        st.dataframe(actual_pred_df.head(20), use_container_width=True)
    else:
        st.warning("Chưa có dữ liệu Actual vs Predicted. Hãy chạy lại train_model.py.")


with tab_comparison:
    st.markdown(
        """
        <div class="section-card">
            <span class="step-badge">Bước 3</span>
            <h3 style="margin-top:0;">So sánh các thuật toán hồi quy</h3>
            <p class="small-note">
                Phần này chứng minh mô hình cuối cùng được chọn dựa trên kết quả đánh giá, không phải chọn cảm tính.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("So sánh thuật toán")

    metrics_df = pd.DataFrame(metadata["metrics"])

    st.subheader("Bảng đánh giá các mô hình")
    st.dataframe(metrics_df, use_container_width=True)

    st.info(
        """
        **Cách đọc bảng:**
        - Mô hình tốt thường có **MAE thấp**, **RMSE thấp** và **R² cao**.
        - RMSE được dùng nhiều khi muốn chú ý đến các lỗi dự đoán lớn.
        - R² giúp đánh giá mức độ mô hình giải thích được biến động của Units Sold.
        """
    )

    st.subheader("So sánh theo R² Score")

    fig_r2 = px.bar(
        metrics_df,
        x="Model",
        y="R2_Score",
        text="R2_Score",
        title="Algorithm Comparison by R² Score",
        labels={
            "Model": "Thuật toán",
            "R2_Score": "R² Score"
        }
    )

    st.plotly_chart(fig_r2, use_container_width=True)

    st.caption("Biểu đồ R² cho biết mô hình nào giải thích dữ liệu tốt hơn. Cột càng cao càng tốt.")

    st.subheader("So sánh theo RMSE")

    fig_rmse = px.bar(
        metrics_df,
        x="Model",
        y="RMSE",
        text="RMSE",
        title="Algorithm Comparison by RMSE",
        labels={
            "Model": "Thuật toán",
            "RMSE": "RMSE"
        }
    )

    st.plotly_chart(fig_rmse, use_container_width=True)

    st.caption("Biểu đồ RMSE cho biết mô hình nào có sai số dự đoán thấp hơn. Cột càng thấp càng tốt.")

    st.success("**Kết luận so sánh thuật toán:** " + algorithm_conclusion(metrics_df, metadata["best_model_name"]))


with tab_explain:
    st.markdown(
        """
        <div class="section-card">
            <span class="step-badge">Bước 4</span>
            <h3 style="margin-top:0;">Giải thích logic dự đoán của mô hình</h3>
            <p class="small-note">
                Feature importance giúp xác định các biến đầu vào mà mô hình dựa vào nhiều nhất khi dự đoán Units Sold.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Giải thích mô hình bằng Feature Importance")

    feature_importance = metadata.get("feature_importance", [])

    if feature_importance:
        feature_df = pd.DataFrame(feature_importance)

        st.subheader("Top biến quan trọng nhất")

        fig_feature = px.bar(
            feature_df.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Feature Importance",
            labels={
                "Importance": "Mức độ quan trọng",
                "Feature": "Biến đầu vào"
            }
        )

        st.plotly_chart(fig_feature, use_container_width=True)

        st.info(
            """
            **Cách đọc biểu đồ:**
            - Trục Y là tên biến đầu vào sau khi mô hình xử lý.
            - Trục X là mức độ quan trọng tương đối của biến.
            - Thanh càng dài nghĩa là biến đó có ảnh hưởng lớn hơn đến kết quả dự đoán.
            - Với biến phân loại đã one-hot encoding, tên biến có thể xuất hiện theo dạng nhóm như `cat__Category_...`.
            """
        )

        st.success("**Kết luận từ Feature Importance:** " + feature_conclusion(feature_df))
        st.dataframe(feature_df, use_container_width=True)

    else:
        st.warning(
            """
            Chưa có Feature Importance. Hãy chạy lại train_model.py, hoặc kiểm tra xem mô hình tốt nhất
            có hỗ trợ thuộc tính feature_importances_ hay không.
            """
        )
