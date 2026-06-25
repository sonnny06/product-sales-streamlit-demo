# Product Sales Forecasting Dashboard

## 1. Giới thiệu dự án

Dự án này xây dựng một ứng dụng Machine Learning dùng để **dự đoán số lượng sản phẩm bán ra (Units Sold)** dựa trên dữ liệu bán lẻ lịch sử.

Mục tiêu chính của dự án là biến dữ liệu kinh doanh thành thông tin có thể hỗ trợ ra quyết định, ví dụ:

- Dự đoán nhu cầu bán hàng.
- Hỗ trợ lập kế hoạch tồn kho.
- Giảm rủi ro thiếu hàng hoặc tồn kho quá nhiều.
- Hỗ trợ quyết định về giá bán, khuyến mãi và marketing.
- Trình bày quy trình Machine Learning dưới dạng web demo dễ sử dụng.

Ứng dụng được triển khai bằng **Streamlit** và mô hình được huấn luyện bằng **scikit-learn**.

---

## 2. Bài toán cần giải quyết

### 2.1. Loại bài toán

Đây là bài toán **Regression (hồi quy)**.

Lý do: biến mục tiêu cần dự đoán là một giá trị số liên tục:

```text
Units Sold
```

Tức là mô hình cần dự đoán một sản phẩm có thể bán được bao nhiêu đơn vị trong một điều kiện kinh doanh cụ thể.

### 2.2. Biến mục tiêu

```text
Units_Sold
```

Biến này thể hiện số lượng sản phẩm đã bán ra.

### 2.3. Dữ liệu đầu vào

Dữ liệu đầu vào gồm các nhóm thông tin chính:

| Nhóm dữ liệu | Ví dụ biến |
|---|---|
| Thông tin cửa hàng | Store ID, Region |
| Thông tin sản phẩm | Product ID, Category |
| Tồn kho và đặt hàng | Inventory Level, Units Ordered |
| Giá và khuyến mãi | Price, Discount, Holiday/Promotion |
| Thị trường | Competitor Pricing |
| Thời gian và mùa vụ | Date, Seasonality |
| Điều kiện bên ngoài | Weather Condition |

---

## 3. Cấu trúc thư mục project

Cấu trúc project đề xuất:

```text
product-sales-streamlit-demo/
│
├── runtime.txt
│
└── product_sales_streamlit_demo/
    │
    ├── app.py
    ├── train_model.py
    ├── requirements.txt
    ├── README.md
    │
    ├── data/
    │   └── retail_store_inventory.csv
    │
    ├── models/
    │   ├── best_product_sales_prediction_model.joblib
    │   └── model_metadata.json
    │
    └── notebook/
        └── rut_gon.ipynb
```

### Ý nghĩa các file chính

| File / thư mục | Ý nghĩa |
|---|---|
| `app.py` | File chạy giao diện web bằng Streamlit |
| `train_model.py` | File huấn luyện, đánh giá và lưu mô hình |
| `requirements.txt` | Danh sách thư viện cần cài |
| `runtime.txt` | Chỉ định phiên bản Python khi deploy |
| `data/retail_store_inventory.csv` | Dataset đầu vào |
| `models/best_product_sales_prediction_model.joblib` | Mô hình tốt nhất đã được lưu |
| `models/model_metadata.json` | Thông tin đánh giá mô hình dùng để hiển thị dashboard |
| `notebook/rut_gon.ipynb` | Notebook gốc, được giữ nguyên để đối chiếu quy trình |

---

## 4. Pipeline Machine Learning

Pipeline tổng quát của dự án:

```text
Raw Data
→ Data Cleaning
→ Feature Engineering
→ Train/Test Split
→ Preprocessing
→ Model Training
→ Model Evaluation
→ Select Best Model
→ Save Model
→ Streamlit Prediction Dashboard
```

---

## 5. Chi tiết từng bước trong pipeline

## 5.1. Load dữ liệu

File dữ liệu được đọc từ:

```text
data/retail_store_inventory.csv
```

Mục đích của bước này là đưa dữ liệu bán lẻ vào chương trình để xử lý và huấn luyện mô hình.

Ví dụ:

```python
df = pd.read_csv(DATA_PATH)
```

---

## 5.2. Làm sạch dữ liệu

Các bước xử lý chính:

### Chuẩn hóa tên cột

Tên cột được đổi về dạng dễ xử lý hơn, ví dụ:

```text
Units Sold → Units_Sold
Inventory Level → Inventory_Level
Competitor Pricing → Competitor_Pricing
```

Mục đích:

- Tránh lỗi khi gọi tên cột trong Python.
- Giúp code nhất quán hơn.
- Dễ dùng trong pipeline và model.

### Xóa dữ liệu trùng

```python
df = df.drop_duplicates()
```

Mục đích:

- Tránh việc một bản ghi bị tính nhiều lần.
- Giảm nhiễu khi huấn luyện mô hình.

### Xóa dòng thiếu biến mục tiêu

```python
df = df.dropna(subset=[target_col])
```

Vì nếu dòng dữ liệu không có `Units_Sold`, mô hình không thể học được đáp án đúng.

### Loại bỏ giá trị bất hợp lý

```python
df = df[df[target_col] >= 0]
```

Vì số lượng bán ra không thể âm.

### Xử lý outlier bằng IQR

Dự án sử dụng phương pháp IQR để loại bỏ các giá trị `Units_Sold` quá bất thường.

Công thức:

```text
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Mục đích:

- Giảm ảnh hưởng của dữ liệu cực đoan.
- Giúp mô hình học ổn định hơn.
- Tránh việc một vài điểm bất thường làm lệch kết quả.

---

## 5.3. Feature Engineering

Feature Engineering là bước tạo thêm biến mới từ dữ liệu ban đầu để mô hình học tốt hơn.

### 5.3.1. Tách đặc trưng thời gian từ Date

Từ cột `Date`, tạo thêm:

| Feature mới | Ý nghĩa |
|---|---|
| Year | Năm |
| Month | Tháng |
| Day | Ngày trong tháng |
| DayOfWeek | Thứ trong tuần |
| Quarter | Quý |
| Is_Weekend | Có phải cuối tuần hay không |

Mục đích:

- Giúp mô hình hiểu yếu tố thời gian.
- Doanh số có thể thay đổi theo tháng, quý, cuối tuần hoặc mùa vụ.
- Tăng khả năng mô hình nhận diện xu hướng bán hàng.

---

### 5.3.2. Discounted Price

```text
Discounted_Price = Price × (1 - Discount / 100)
```

Ý nghĩa:

- Giá thực tế sau khi giảm giá.
- Có thể ảnh hưởng trực tiếp đến nhu cầu mua hàng.

Ví dụ: nếu giá gốc là 100 và giảm giá 20%, giá sau giảm là 80.

---

### 5.3.3. Price Difference

```text
Price_Difference = Price - Competitor_Pricing
```

Ý nghĩa:

- So sánh giá bán của cửa hàng với giá của đối thủ.
- Nếu giá cao hơn đối thủ quá nhiều, doanh số có thể giảm.
- Nếu giá cạnh tranh hơn, doanh số có thể tăng.

---

### 5.3.4. Inventory Level Log

```text
Inventory_Level_Log = log(1 + Inventory_Level)
```

Ý nghĩa:

- Giảm ảnh hưởng của các giá trị tồn kho quá lớn.
- Giúp dữ liệu ổn định hơn.
- Hỗ trợ một số mô hình học tốt hơn với dữ liệu có độ lệch lớn.

---

## 5.4. Loại bỏ leakage column

Cột `Demand_Forecast` được loại bỏ khỏi dữ liệu huấn luyện.

Lý do:

- `Demand_Forecast` có thể đã là một giá trị dự báo sẵn.
- Nếu dùng cột này để dự đoán `Units_Sold`, mô hình có thể bị “ăn gian”.
- Điều này làm kết quả đánh giá có vẻ rất tốt nhưng không phản ánh đúng năng lực thật của mô hình.

Đây gọi là hiện tượng **data leakage**.

---

## 5.5. Chia dữ liệu train/test

Dữ liệu được chia thành:

```text
80% training set
20% testing set
```

Mục đích:

- Training set dùng để huấn luyện mô hình.
- Testing set dùng để kiểm tra mô hình trên dữ liệu chưa từng thấy.
- Giúp đánh giá khả năng tổng quát hóa của mô hình.

---

## 5.6. Tiền xử lý dữ liệu

Dữ liệu có 2 loại chính:

```text
Numeric features
Categorical features
```

### Numeric features

Các biến số như:

```text
Inventory_Level, Units_Ordered, Price, Discount, Competitor_Pricing
```

Được xử lý bằng:

```text
SimpleImputer(strategy="median")
StandardScaler()
```

Ý nghĩa:

- `SimpleImputer`: thay giá trị thiếu bằng trung vị.
- `StandardScaler`: chuẩn hóa dữ liệu số để mô hình học ổn định hơn.

### Categorical features

Các biến phân loại như:

```text
Store_ID, Product_ID, Category, Region, Weather_Condition, Seasonality
```

Được xử lý bằng:

```text
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

Ý nghĩa:

- Thay giá trị thiếu bằng giá trị xuất hiện nhiều nhất.
- One-hot encoding chuyển dữ liệu dạng chữ thành dạng số để mô hình có thể học được.

---

## 6. Các thuật toán được sử dụng

Dự án huấn luyện và so sánh nhiều mô hình hồi quy:

| Thuật toán | Vai trò |
|---|---|
| Linear Regression | Mô hình baseline đơn giản |
| Decision Tree Regressor | Mô hình cây quyết định |
| Random Forest Regressor | Mô hình ensemble nhiều cây |
| Gradient Boosting Regressor | Mô hình boosting |
| Tuned Random Forest | Random Forest được tối ưu tham số bằng GridSearchCV |

---

## 6.1. Linear Regression

Linear Regression là mô hình tuyến tính, giả định mối quan hệ giữa biến đầu vào và `Units_Sold` là dạng tuyến tính.

Ưu điểm:

- Dễ hiểu.
- Dễ giải thích.
- Phù hợp làm baseline.

Nhược điểm:

- Không mạnh với dữ liệu phi tuyến.
- Khó bắt được các quan hệ phức tạp.

---

## 6.2. Decision Tree Regressor

Decision Tree học bằng cách chia dữ liệu thành nhiều nhánh quyết định.

Ưu điểm:

- Dễ hình dung.
- Có thể học quan hệ phi tuyến.
- Không yêu cầu dữ liệu phải tuyến tính.

Nhược điểm:

- Dễ overfitting nếu cây quá sâu.
- Nhạy với dữ liệu nhiễu.

---

## 6.3. Random Forest Regressor

Random Forest là tập hợp nhiều Decision Tree.

Ưu điểm:

- Ổn định hơn một cây đơn lẻ.
- Giảm overfitting.
- Thường cho kết quả tốt với dữ liệu dạng bảng.

Nhược điểm:

- Khó giải thích hơn Linear Regression.
- Tốn tài nguyên hơn.

---

## 6.4. Gradient Boosting Regressor

Gradient Boosting xây dựng nhiều mô hình yếu theo tuần tự, mô hình sau sửa lỗi cho mô hình trước.

Ưu điểm:

- Hiệu quả cao với nhiều bài toán dữ liệu bảng.
- Có khả năng học quan hệ phức tạp.

Nhược điểm:

- Có thể nhạy với tham số.
- Train lâu hơn một số mô hình đơn giản.

---

## 6.5. Tuned Random Forest

Random Forest được tối ưu bằng GridSearchCV.

GridSearchCV thử nhiều tổ hợp tham số khác nhau, ví dụ:

```text
n_estimators
max_depth
min_samples_split
```

Mục đích:

- Tìm bộ tham số tốt hơn.
- Cải thiện độ chính xác.
- Chọn mô hình cuối cùng có hiệu năng tốt hơn.

---

## 7. Đánh giá mô hình

Các chỉ số đánh giá chính:

| Chỉ số | Ý nghĩa | Tốt khi |
|---|---|---|
| MAE | Sai số tuyệt đối trung bình | Càng thấp càng tốt |
| MSE | Sai số bình phương trung bình | Càng thấp càng tốt |
| RMSE | Căn bậc hai của MSE | Càng thấp càng tốt |
| R² Score | Mức độ mô hình giải thích dữ liệu | Càng cao càng tốt |

---

## 7.1. MAE

```text
MAE = trung bình của |giá trị thật - giá trị dự đoán|
```

Ý nghĩa:

- Cho biết trung bình mô hình dự đoán lệch bao nhiêu đơn vị.
- Dễ giải thích nhất với người không chuyên.

Ví dụ:

```text
MAE = 12
```

Có nghĩa là trung bình mô hình dự đoán lệch khoảng 12 sản phẩm.

---

## 7.2. RMSE

```text
RMSE = căn bậc hai của MSE
```

Ý nghĩa:

- Cũng đo sai số dự đoán.
- Phạt mạnh hơn với các lỗi lớn.
- Nếu RMSE cao hơn MAE nhiều, có thể có một số dự đoán bị lệch mạnh.

---

## 7.3. R² Score

```text
R² Score = mức độ mô hình giải thích được biến động của dữ liệu
```

Ý nghĩa:

- R² càng gần 1 thì mô hình càng giải thích dữ liệu tốt.
- R² thấp cho thấy mô hình chưa học tốt hoặc dữ liệu còn nhiều nhiễu.

Ví dụ:

```text
R² = 0.85
```

Có thể hiểu là mô hình giải thích được khoảng 85% biến động trong dữ liệu kiểm tra.

---

## 8. File model_metadata.json

Sau khi train, hệ thống tạo file:

```text
models/model_metadata.json
```

File này chứa thông tin phục vụ dashboard:

| Thành phần | Ý nghĩa |
|---|---|
| best_model_name | Tên mô hình tốt nhất |
| metrics | Bảng so sánh các mô hình |
| best_model_metrics | Chỉ số của mô hình tốt nhất |
| actual_vs_predicted | Dữ liệu để vẽ biểu đồ giá trị thật và dự đoán |
| feature_importance | Các biến ảnh hưởng mạnh đến mô hình |
| category_options | Danh sách giá trị cho các ô chọn trong app |
| default_values | Giá trị mặc định cho input |

---

## 9. Ứng dụng Streamlit

Ứng dụng gồm 4 tab chính:

```text
Dự đoán
Độ tin cậy mô hình
So sánh thuật toán
Giải thích mô hình
```

---

## 9.1. Tab Dự đoán

Người dùng nhập thông tin:

- Ngày.
- Store ID.
- Product ID.
- Category.
- Region.
- Inventory Level.
- Units Ordered.
- Price.
- Discount.
- Weather Condition.
- Holiday/Promotion.
- Competitor Pricing.
- Seasonality.

Sau đó ứng dụng tạo các feature mới và dự đoán:

```text
Predicted Units Sold
```

Ứng dụng cũng đưa ra:

- Mức nhu cầu: thấp, trung bình, cao.
- Gợi ý kinh doanh dựa trên kết quả dự đoán.

---

## 9.2. Tab Độ tin cậy mô hình

Tab này hiển thị:

- MAE.
- RMSE.
- R² Score.
- Biểu đồ Actual vs Predicted.

### Ý nghĩa biểu đồ Actual vs Predicted

Biểu đồ này so sánh:

```text
Giá trị thực tế vs Giá trị dự đoán
```

Cách đọc:

- Mỗi điểm là một dòng dữ liệu test.
- Trục X là `Units Sold` thực tế.
- Trục Y là `Units Sold` mô hình dự đoán.
- Nếu điểm càng gần đường xu hướng, mô hình dự đoán càng sát.
- Nếu điểm nằm xa, mô hình dự đoán lệch nhiều.

Kết luận rút ra:

- Nếu các điểm phân bố gần xu hướng chéo, mô hình có độ tin cậy tốt hơn.
- Nếu các điểm phân tán quá rộng, mô hình cần được cải thiện thêm.

---

## 9.3. Tab So sánh thuật toán

Tab này hiển thị:

- Bảng so sánh các mô hình.
- Biểu đồ so sánh R².
- Biểu đồ so sánh RMSE.

### Ý nghĩa biểu đồ R²

- Cột càng cao thì mô hình giải thích dữ liệu càng tốt.
- Dùng để xác định mô hình nào học được quan hệ giữa input và `Units_Sold` tốt hơn.

### Ý nghĩa biểu đồ RMSE

- Cột càng thấp thì sai số dự đoán càng thấp.
- Dùng để xác định mô hình nào ít dự đoán lệch hơn.

Kết luận rút ra:

- Mô hình cuối cùng nên là mô hình có RMSE thấp và R² cao.
- Nếu Random Forest hoặc Tuned Random Forest có kết quả tốt nhất, điều này cho thấy mô hình ensemble phù hợp với dữ liệu bán lẻ có quan hệ phi tuyến.

---

## 9.4. Tab Giải thích mô hình

Tab này hiển thị biểu đồ Feature Importance.

### Ý nghĩa Feature Importance

Feature Importance cho biết biến nào được mô hình sử dụng nhiều nhất khi dự đoán.

Cách đọc:

- Thanh càng dài, biến càng quan trọng.
- Biến quan trọng hơn có ảnh hưởng lớn hơn đến kết quả dự đoán.
- Với biến phân loại, tên biến có thể xuất hiện dưới dạng sau one-hot encoding, ví dụ:

```text
cat__Category_Clothing
cat__Region_East
```

Kết luận rút ra:

- Nếu các biến như `Inventory_Level`, `Price`, `Discount`, `Competitor_Pricing` có importance cao, điều đó hợp lý vì đây là các yếu tố ảnh hưởng trực tiếp đến nhu cầu bán hàng.
- Feature importance giúp giải thích vì sao mô hình đưa ra dự đoán, nhưng không khẳng định quan hệ nhân quả tuyệt đối.

---

## 10. Cách chạy project trên máy local

### Bước 1: Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu bị lỗi với scikit-learn do Python 3.14, có thể dùng:

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib plotly
```

---

### Bước 2: Train lại mô hình

```bash
cd product_sales_streamlit_demo
python train_model.py
```

Sau khi chạy xong, kiểm tra thư mục:

```text
models/
```

Phải có:

```text
best_product_sales_prediction_model.joblib
model_metadata.json
```

---

### Bước 3: Chạy Streamlit app

```bash
streamlit run app.py
```

Sau đó mở đường dẫn local do Streamlit cung cấp.

---


## 11. Ý nghĩa thực tiễn của dự án

Dự án cho thấy Machine Learning có thể hỗ trợ doanh nghiệp bán lẻ trong các quyết định:

| Vấn đề kinh doanh | Cách mô hình hỗ trợ |
|---|---|
| Tồn kho quá nhiều | Dự đoán nhu cầu thấp để tránh nhập dư |
| Thiếu hàng | Dự đoán nhu cầu cao để tăng tồn kho |
| Khuyến mãi | Xem ảnh hưởng của discount và promotion |
| Cạnh tranh giá | Xem chênh lệch giá với đối thủ |
| Mùa vụ | Xem ảnh hưởng của seasonality |
| Ra quyết định | Dựa trên dữ liệu thay vì cảm tính |

---

## 12. Kết luận

Dự án này xây dựng hoàn chỉnh quy trình Machine Learning cho bài toán dự báo doanh số sản phẩm:

```text
Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

Mô hình cuối cùng không chỉ đưa ra kết quả `Predicted Units Sold`, mà còn được giải thích bằng:

- Chỉ số MAE, RMSE, R².
- So sánh nhiều thuật toán.
- Biểu đồ Actual vs Predicted.
- Feature Importance.
- Gợi ý kinh doanh dựa trên kết quả dự đoán.

Điều này giúp ứng dụng không chỉ là một demo dự đoán đơn giản, mà trở thành một dashboard Machine Learning có khả năng hỗ trợ ra quyết định kinh doanh.
