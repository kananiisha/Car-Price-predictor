import streamlit as st
import joblib
import numpy as np
import random

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PREDICTION_INTERVAL_PCT = 0.295

@st.cache_resource
def load_artifacts():
    model     = joblib.load("car_price_model.pkl")
    le_fuel   = joblib.load("encoders/le_fuel.pkl")
    le_seller = joblib.load("encoders/le_seller.pkl")
    le_trans  = joblib.load("encoders/le_transmission.pkl")
    return model, le_fuel, le_seller, le_trans

model, le_fuel, le_seller, le_trans = load_artifacts()

bg_images = [
    "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1400&q=80",
    "https://images.unsplash.com/photo-1511919884226-fd3cad34687c?auto=format&fit=crop&w=1400&q=80",
    "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?auto=format&fit=crop&w=1400&q=80",
]
bg = random.choice(bg_images)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* {{ font-family: 'Inter', sans-serif; }}
.stApp {{
    background: linear-gradient(160deg, rgba(5,12,29,0.94), rgba(10,30,60,0.88)),
                url('{bg}') center/cover fixed;
}}
[data-testid="stDecoration"],[data-testid="stToolbar"],[data-testid="stHeader"] {{ display: none !important; }}
#MainMenu, footer {{ visibility: hidden !important; }}
.block-container {{ padding: 2rem 3rem; max-width: 1150px; }}
label, p, .stMarkdown p {{ color: #d0e4ff !important; }}
[data-testid="stNumberInput"] > div {{
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 8px !important; overflow: hidden !important;
}}
[data-testid="stNumberInput"] div {{ background-color: transparent !important; border: none !important; box-shadow: none !important; }}
[data-testid="stNumberInput"] input {{ background-color: transparent !important; color: #ffffff !important; border: none !important; box-shadow: none !important; }}
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {{
    background-color: rgba(255,255,255,0.08) !important; color: #fff !important;
    border: none !important; border-left: 1px solid rgba(255,255,255,0.12) !important;
}}
[data-baseweb="select"] > div {{
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.18) !important; border-radius: 8px !important;
}}
[data-baseweb="select"] div {{ background-color: transparent !important; border: none !important; }}
[data-baseweb="select"] span, [data-baseweb="select"] div {{ color: #ffffff !important; }}
[data-baseweb="select"] svg {{ fill: #ffffff !important; }}
[data-baseweb="popover"] *, [data-baseweb="menu"] *, ul[role="listbox"], ul[role="listbox"] * {{
    background-color: rgba(8,22,52,0.98) !important; color: #d0e4ff !important;
    border-color: rgba(255,255,255,0.12) !important;
}}
li[role="option"]:hover, [aria-selected="true"] {{ background-color: rgba(15,76,129,0.5) !important; color: #ffffff !important; }}
.stButton > button {{
    background: linear-gradient(135deg, #0f4c81, #0a7c5c) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    font-size: 1rem !important; font-weight: 600 !important; padding: 0.75rem 2rem !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; }}
</style>
""", unsafe_allow_html=True)

CARD  = "background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.13);border-radius:16px;padding:1.4rem 1.6rem;margin-bottom:1.1rem;"
TITLE = "color:rgba(130,190,255,0.9);font-size:0.75rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;margin:0 0 0.9rem 0;"
ROW   = "background:rgba(255,255,255,0.05);border-left:3px solid rgba(15,76,129,0.9);border-radius:0 10px 10px 0;padding:0.6rem 0.9rem;margin:0.45rem 0;color:#c8dcff;font-size:0.88rem;"

st.markdown("""
<div style="background:linear-gradient(135deg,rgba(15,76,129,0.6),rgba(0,180,120,0.25));
            border:1px solid rgba(255,255,255,0.12);border-radius:20px;
            padding:2rem 2.5rem;margin-bottom:1.8rem;text-align:center;">
  <h1 style="color:#fff;font-size:2.3rem;font-weight:700;margin:0 0 0.3rem 0;">🚗 Car Price Predictor</h1>
  <p style="color:rgba(200,220,255,0.85);font-size:1rem;margin:0;">
    Enter your car details to get an instant resale value estimate — powered by Gradient Boosting ML
  </p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown(f'<p style="{TITLE}">📋 Car Details</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        present_price = st.number_input("Showroom Price (Rs. Lakhs)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
        fuel_type     = st.selectbox("Fuel Type", le_fuel.classes_.tolist())
        owner         = st.selectbox("Previous Owners", [0, 1, 2, 3],
                                     format_func=lambda x: "First Owner" if x == 0 else f"{x} Owner{'s' if x > 1 else ''}")
    with c2:
        driven_kms   = st.number_input("Kilometers Driven", min_value=0, max_value=500_000, value=30_000, step=1_000)
        transmission = st.selectbox("Transmission", le_trans.classes_.tolist())
        selling_type = st.selectbox("Seller Type", le_seller.classes_.tolist())
    car_age = st.slider("Car Age (years)", min_value=0, max_value=20, value=5)
    predict_clicked = st.button("🔍 Predict Resale Price", use_container_width=True)

with col_right:
    st.markdown(f"""
    <div style="{CARD}">
      <p style="{TITLE}">💡 Market Insights</p>
      <div style="{ROW}">🏆 <b>Diesel cars</b> hold resale value better than Petrol</div>
      <div style="{ROW}">⚙️ <b>Automatic</b> transmission commands a price premium</div>
      <div style="{ROW}">📅 Value drops <b>significantly after 5 years</b> of ownership</div>
      <div style="{ROW}">📍 <b>Low mileage</b> is one of the strongest price boosters</div>
      <div style="{ROW}">🤝 <b>Dealer-sold</b> cars typically fetch higher prices</div>
    </div>
    """, unsafe_allow_html=True)

def validate_inputs(present_price, driven_kms, car_age):
    warnings = []
    if car_age > 10 and driven_kms < 10_000:
        warnings.append("⚠️ Very low mileage for a car this old — double-check kilometers.")
    if present_price < 1.0 and car_age == 0:
        warnings.append("⚠️ Showroom price seems low for a brand-new car.")
    if driven_kms > 300_000:
        warnings.append("⚠️ Extremely high mileage — prediction accuracy may reduce.")
    return warnings

if predict_clicked:
    for w in validate_inputs(present_price, driven_kms, car_age):
        st.warning(w)

    fuel_enc  = le_fuel.transform([fuel_type])[0]
    sell_enc  = le_seller.transform([selling_type])[0]
    trans_enc = le_trans.transform([transmission])[0]
    features  = np.array([[present_price, driven_kms, fuel_enc, sell_enc, trans_enc, owner, car_age]])
    prediction   = max(0.1, model.predict(features)[0])
    low          = prediction * (1 - PREDICTION_INTERVAL_PCT)
    high         = prediction * (1 + PREDICTION_INTERVAL_PCT)
    depreciation = round((1 - prediction / present_price) * 100, 1) if present_price > 0 else 0
    interval_display = int(PREDICTION_INTERVAL_PCT * 100)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,200,120,0.13),rgba(0,120,200,0.13));
                border:2px solid rgba(0,210,130,0.45);border-radius:18px;
                padding:2rem;text-align:center;margin:1.5rem 0 1rem 0;">
      <p style="color:rgba(180,220,255,0.8);font-size:0.85rem;font-weight:600;
                letter-spacing:1.2px;text-transform:uppercase;margin:0 0 0.4rem 0;">Estimated Resale Value</p>
      <p style="color:#00e896;font-size:3rem;font-weight:700;margin:0.2rem 0;">Rs. {prediction:.2f} Lakhs</p>
      <p style="color:rgba(200,220,255,0.6);font-size:0.88rem;margin:0.4rem 0 0 0;">
        90% confidence range (±{interval_display}%):&nbsp; Rs. {low:.2f} – Rs. {high:.2f} Lakhs
      </p>
    </div>
    """, unsafe_allow_html=True)

    def pill(bg, border, color, text):
        return (f'<span style="background:{bg};border:1px solid {border};color:{color};'
                f'padding:0.28rem 0.75rem;border-radius:20px;font-size:0.81rem;'
                f'display:inline-block;margin:0.2rem;">{text}</span>')

    G = ("rgba(0,200,100,0.15)",  "rgba(0,200,100,0.4)",  "#00e896")
    N = ("rgba(255,200,60,0.15)", "rgba(255,200,60,0.4)", "#ffd060")
    B = ("rgba(255,100,80,0.15)", "rgba(255,100,80,0.4)", "#ff7f6e")

    pills  = (pill(*G, "Diesel – holds value well")   if fuel_type == "Diesel"
         else pill(*N, "Petrol – average resale")      if fuel_type == "Petrol"
         else pill(*B, "CNG – lower resale demand"))
    pills += (pill(*G, "Automatic – premium pricing") if transmission == "Automatic"
         else pill(*N, "Manual – standard pricing"))
    pills += (pill(*G, "Low age – minimal depreciation")     if car_age <= 3
         else pill(*N, "Moderate age – normal wear")          if car_age <= 7
         else pill(*B, "High age – significant depreciation"))
    pills += (pill(*G, "Low mileage – great resale")    if driven_kms < 30_000
         else pill(*N, "Moderate mileage – acceptable") if driven_kms < 80_000
         else pill(*B, "High mileage – reduces value"))
    pills += (pill(*G, "First owner – highest trust")   if owner == 0
         else pill(*N, "Second owner – minor impact")    if owner == 1
         else pill(*B, "Multiple owners – lower demand"))
    pills += (pill(*G, "Dealer sale – trust premium")   if selling_type == "Dealer"
         else pill(*N, "Individual sale – negotiable"))

    st.markdown(f"""
    <div style="{CARD}">
      <p style="{TITLE}">Price Factors for Your Car</p>
      <div style="line-height:2.4;">{pills}</div>
      <div style="margin-top:1rem;padding-top:0.8rem;
                  border-top:1px solid rgba(255,255,255,0.08);
                  color:rgba(200,220,255,0.65);font-size:0.86rem;">
        Depreciated approximately <b style="color:#ffd060;">{depreciation}%</b>
        from showroom price of Rs. {present_price:.2f} Lakhs
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:3rem;color:rgba(180,200,240,0.35);font-size:0.78rem;">
  Built with Streamlit &middot; Gradient Boosting ML &middot; scikit-learn &middot;
  <a href="https://github.com/kananiisha/Car-Price-predictor" style="color:rgba(130,190,255,0.4);text-decoration:none;">GitHub</a>
</div>
""", unsafe_allow_html=True)