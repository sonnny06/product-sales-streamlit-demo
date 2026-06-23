# Product Sales Prediction Streamlit Demo

This project is a small web demo for the university machine learning topic **Predicting Product Sales**.

## Important note about `rut_gon.ipynb`

The original notebook is kept **100% unchanged** in:

```text
notebook/rut_gon.ipynb
```

The file `train_model.py` follows the same machine learning workflow from `rut_gon.ipynb`, including:

- data loading
- column cleaning
- duplicate removal
- missing target removal
- invalid target removal
- IQR outlier handling
- date feature engineering
- `Discounted_Price`
- `Price_Difference`
- `Inventory_Level_Log`
- dropping `Demand_Forecast`
- preprocessing with `ColumnTransformer`
- model comparison
- `GridSearchCV` tuned Random Forest
- saving the best model with `joblib`

Only notebook-only parts such as EDA plots and display cells are removed from `train_model.py` so the script can run as a normal Python training file.

One small compatibility fix is added in `train_model.py`: the notebook saves `final_model`, but `final_model` is not explicitly assigned in the final notebook cell. In the script, `final_model` is assigned from the best model in `final_results_df` so the web app can load the trained model reliably.

## Project structure

```text
product_sales_streamlit_demo/
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

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to retrain the model

```bash
python train_model.py
```

After retraining, run the app again:

```bash
streamlit run app.py
```

## Machine learning flow

```text
Historical Retail Data
→ Data Cleaning
→ Feature Engineering
→ Preprocessing
→ Regression Model Training
→ Best Model Selection
→ Predicted Units Sold
→ Business Recommendation
```
