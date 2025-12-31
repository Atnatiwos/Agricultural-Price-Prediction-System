import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import calendar

# Load model and features
#model = joblib.load("price_prediction_model.pkl")
#features = joblib.load("model_features.pkl")

model = joblib.load("/content/drive/MyDrive/Colab/agriculture_project/Model/price_model.pkl")
features = joblib.load("/content/drive/MyDrive/Colab/agriculture_project/Model/model_features.pkl")

st.set_page_config(
    page_title="Ethiopian Agricultural Price Predictor",
    layout="wide"
)

# ---------------- HEADER ----------------
st.title("🌾 Ethiopian Agricultural Price Prediction System")
st.markdown(
    "Predict **next month agricultural commodity prices** in Ethiopian markets using machine learning."
)
st.divider()

# ---------------- INPUTS ----------------
st.subheader("📥 Prediction Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    commodity = st.selectbox(
        "Commodity",
        ["Teff (white)", "Wheat (white)", "Maize (white)", "Sorghum (white)", "Barley"],
        help="Select the commodity to forecast"
    )

with col2:
    market = st.selectbox(
        "Market",
        ["Addis Ababa", "Adama", "Bahir Dar", "Hawassa", "Jimma", "Dire Dawa"],
        help="Select the market location"
    )

with col3:
    month = st.selectbox(
        "Prediction Month",
        list(range(1,13)),
        help="Select the current month; the system predicts the NEXT month price"
    )

col4, col5, col6 = st.columns(3)

with col4:
    prev_price = st.number_input(
        "Previous Month Price (ETB/kg)",
        5.0, 150.0, 40.0,
        help="Observed price in the most recent month"
    )

with col5:
    ma3 = st.number_input(
        "3-Month Moving Average (ETB/kg)",
        5.0, 150.0, 38.0,
        help="Average price of the last three months"
    )

with col6:
    rainfall = st.slider(
        "Rainfall Index",
        1, 10, 5,
        help="Seasonal rainfall intensity affecting crop supply"
    )

# ---------------- AUTO HOLIDAY & FASTING ----------------
def holiday_fasting_flags(month):
    holiday_months = [4, 9, 12]   # Easter, Meskel, Christmas/Eid
    fasting_months = [3, 4, 6]    # Lent, Ramadan (approx)
    return {
        "Holiday": 1 if month in holiday_months else 0,
        "Fasting": 1 if month in fasting_months else 0
    }

flags = holiday_fasting_flags(month)

# ---------------- SEASON FLAGS ----------------
def season_flags(month):
    if month in [2,3,4,5]:
        return {"Season_Belg":1, "Season_Meher":0}
    elif month in [6,7,8,9]:
        return {"Season_Belg":0, "Season_Meher":1}
    return {"Season_Belg":0, "Season_Meher":0}

season = season_flags(month)

# ---------------- BUILD INPUT ----------------
data = {
    "Month": month,
    "Rainfall_Index": rainfall,
    "Holiday": flags["Holiday"],
    "Fasting": flags["Fasting"],
    "Prev_Price": prev_price,
    "MA_3": ma3,
    "Season_Belg": season["Season_Belg"],
    "Season_Meher": season["Season_Meher"],
}

for col in features:
    if col.startswith("commodity_"):
        data[col] = 1 if col == f"commodity_{commodity}" else 0
    if col.startswith("market_"):
        data[col] = 1 if col == f"market_{market}" else 0
    if col.startswith("pricetype_"):
        data[col] = 1

input_df = pd.DataFrame([data])

for col in features:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[features]

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Next Month Price", use_container_width=True):

    prediction = model.predict(input_df)[0]

    next_month = 1 if month == 12 else month + 1
    next_month_name = calendar.month_name[next_month]

    st.success(
        f"💰 Predicted Price for **{next_month_name}**: {prediction:.2f} ETB per KG"
    )

    # ---------------- 3-MONTH FORECAST ----------------
    st.subheader("📈 3-Month Price Forecast")

    preds = []
    prev = prev_price
    avg = ma3
    m = month

    for _ in range(3):
        input_df["Prev_Price"] = prev
        input_df["MA_3"] = (prev + avg) / 2
        p = model.predict(input_df)[0]
        preds.append(p)
        prev = p
        m = 1 if m == 12 else m + 1

    months = [calendar.month_name[(month+i-1)%12+1] for i in range(1,4)]

    for mo, pr in zip(months, preds):
        st.write(f"📅 {mo}: **{pr:.2f} ETB/kg**")

    # ---------------- TREND CHART ----------------
    plt.figure()
    plt.plot(months, preds, marker="o")
    plt.xlabel("Month")
    plt.ylabel("Price (ETB/kg)")
    plt.title("3-Month Price Forecast Trend")
    plt.grid(True)
    st.pyplot(plt)

st.divider()
st.caption("Machine Learning Project – Ethiopian Agricultural Markets")
