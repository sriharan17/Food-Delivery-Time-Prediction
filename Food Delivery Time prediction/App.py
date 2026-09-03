from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

px.defaults.template = "plotly_dark"

st.set_page_config(page_title="Dispatch | Delivery time intelligence", page_icon="FD", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#edf5ee; --muted:#9aaea2; --green:#63d39a; --orange:#f5a064; --paper:#101815; --panel:#17221d; --line:#2b3b32; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: radial-gradient(circle at 92% 2%, #193b2c 0, transparent 27%), var(--paper); }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
h1 { font-size: clamp(2rem, 4vw, 3.5rem) !important; line-height: 1.02 !important; }
.eyebrow { color:var(--green); font-size:.76rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.lede { color:var(--muted); font-size:1.03rem; max-width:650px; margin-bottom:1.5rem; }
.metric-card { background:rgba(23,34,29,.88); border:1px solid var(--line); border-radius:10px; padding:1.05rem 1.2rem; min-height:112px; box-shadow:0 8px 22px rgba(0,0,0,.18); }
.metric-label { color:var(--muted); font-size:.8rem; font-weight:600; text-transform:uppercase; letter-spacing:.07em; }
.metric-value { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; margin-top:.35rem; }
.metric-note { color:var(--green); font-size:.78rem; margin-top:.2rem; }
.section-rule { border-top:1px solid var(--line); margin:1.8rem 0 1.1rem; }
[data-testid="stSidebar"] { background:#141f1a; border-right:1px solid var(--line); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption { color:var(--muted); }
input, textarea, [data-baseweb="select"] > div { background-color:#202d26 !important; color:var(--ink) !important; border-color:var(--line) !important; }
.stButton > button { background:var(--green); color:#102017; border:0; border-radius:7px; font-weight:700; width:100%; }
.stButton > button:hover { background:#8ce5b4; color:#102017; }
[data-testid="stDataFrame"] { border:1px solid var(--line); }
[data-testid="stMetric"] { background:var(--panel); border-color:var(--line); }
.stTabs [data-baseweb="tab-list"] { gap:1.5rem; }
.stTabs [data-baseweb="tab"] { color:var(--muted); }
.stTabs [aria-selected="true"] { color:var(--green) !important; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).with_name("Food_Delivery_Times.csv")
FEATURES = ["Distance_km", "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"]
TARGET = "Delivery_Time_min"


@st.cache_data
def load_data():
	data = pd.read_csv(DATA_PATH)
	for column in ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs", TARGET]:
		data[column] = pd.to_numeric(data[column], errors="coerce")
	return data


@st.cache_resource
def train_model(data):
	clean_data = data.dropna(subset=[TARGET]).copy()
	x_data, y_data = clean_data[FEATURES], clean_data[TARGET]
	categorical = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
	numeric = [column for column in FEATURES if column not in categorical]
	preprocessor = ColumnTransformer(transformers=[
		("numeric", SimpleImputer(strategy="median"), numeric),
		("categorical", Pipeline(steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", OneHotEncoder(handle_unknown="ignore")),
		]), categorical),
	])
	model = Pipeline(steps=[
		("preprocessor", preprocessor),
		("regressor", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1)),
	])
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.2, random_state=42)
	model.fit(x_train, y_train)
	predictions = model.predict(x_test)
	return model, {"mae": mean_absolute_error(y_test, predictions), "r2": r2_score(y_test, predictions), "actual": y_test, "predicted": predictions}


data = load_data()
model, evaluation = train_model(data)

st.markdown('<div class="eyebrow">Dispatch intelligence / live model</div>', unsafe_allow_html=True)
st.title("Know the arrival before it happens.")
st.markdown('<div class="lede">A practical command center for estimating delivery time, spotting operational drag, and making faster dispatch decisions.</div>', unsafe_allow_html=True)

with st.sidebar:
	st.markdown("## Dispatch console")
	st.caption("Prediction inputs")
	distance = st.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=8.0, step=0.1)
	preparation = st.number_input("Preparation time (min)", min_value=0, max_value=180, value=18, step=1)
	experience = st.number_input("Courier experience (years)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
	weather = st.selectbox("Weather", sorted(data["Weather"].dropna().unique()))
	traffic = st.selectbox("Traffic level", sorted(data["Traffic_Level"].dropna().unique()))
	time_of_day = st.selectbox("Time of day", sorted(data["Time_of_Day"].dropna().unique()))
	vehicle = st.selectbox("Vehicle type", sorted(data["Vehicle_Type"].dropna().unique()))
	predict_clicked = st.button("Estimate delivery time")
	st.markdown("---")
	st.caption(f"Source: {DATA_PATH.name}\n\nRows loaded: {len(data):,}")

if predict_clicked or "prediction" not in st.session_state:
	input_row = pd.DataFrame([{
		"Distance_km": distance, "Weather": weather, "Traffic_Level": traffic,
		"Time_of_Day": time_of_day, "Vehicle_Type": vehicle,
		"Preparation_Time_min": preparation, "Courier_Experience_yrs": experience,
	}])
	st.session_state.prediction = float(model.predict(input_row)[0])

prediction = st.session_state.prediction
st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
metric_columns = st.columns(4)
metric_cards = [
	("Estimated arrival", f"{prediction:.0f} min", f"about {prediction / 60:.1f} hours"),
	("Average delivery", f"{data[TARGET].mean():.0f} min", "across all recorded orders"),
	("Typical model error", f"{evaluation['mae']:.1f} min", "mean absolute error"),
	("On-time signal", f"{evaluation['r2']:.0%}", "variance explained by model"),
]
for column, (label, value, note) in zip(metric_columns, metric_cards):
	with column:
		st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
tab_predict, tab_operations, tab_model = st.tabs(["Prediction detail", "Operations", "Model health"])

with tab_predict:
	left, right = st.columns([1, 1.35], gap="large")
	with left:
		st.subheader("Current estimate")
		st.metric("Delivery time", f"{prediction:.0f} minutes")
		st.caption("The estimate combines route distance, kitchen preparation, traffic, conditions, time of day, vehicle, and courier experience.")
		factors = pd.DataFrame({"Factor": ["Distance", "Preparation", "Courier experience"], "Value": [distance, preparation, experience]})
		st.dataframe(factors, hide_index=True, use_container_width=True)
	with right:
		st.subheader("Where this order sits")
		distribution = px.histogram(data, x=TARGET, nbins=30, color_discrete_sequence=["#1f7a52"], labels={TARGET: "Delivery time (minutes)"})
		distribution.add_vline(x=prediction, line_dash="dash", line_color="#ee8a45", annotation_text="your estimate")
		distribution.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
		st.plotly_chart(distribution, use_container_width=True)

with tab_operations:
	st.subheader("Delivery patterns")
	chart_left, chart_right = st.columns(2)
	with chart_left:
		grouped = data.groupby("Traffic_Level", dropna=False, as_index=False)[TARGET].mean().sort_values(TARGET)
		chart = px.bar(grouped, x="Traffic_Level", y=TARGET, color=TARGET, color_continuous_scale=["#d9f5df", "#1f7a52"], labels={TARGET: "Average minutes"})
		chart.update_layout(height=330, margin=dict(l=0, r=0, t=30, b=0), coloraxis_showscale=False, title="Traffic impact")
		st.plotly_chart(chart, use_container_width=True)
	with chart_right:
		vehicle_summary = data.groupby("Vehicle_Type", as_index=False).agg(average_time=(TARGET, "mean"), orders=(TARGET, "size"))
		chart = px.scatter(vehicle_summary, x="Vehicle_Type", y="average_time", size="orders", color="Vehicle_Type", labels={"average_time": "Average minutes"})
		chart.update_layout(height=330, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, title="Vehicle mix")
		st.plotly_chart(chart, use_container_width=True)
	st.subheader("Recent delivery records")
	st.dataframe(data.sort_values("Order_ID", ascending=False).head(12), hide_index=True, use_container_width=True)

with tab_model:
	st.subheader("Validation snapshot")
	st.caption("The model is evaluated on a held-out 20% sample that was not used during training.")
	actual = pd.DataFrame({"Actual": evaluation["actual"].values, "Predicted": evaluation["predicted"]})
	scatter = px.scatter(actual, x="Actual", y="Predicted", color_discrete_sequence=["#ee8a45"], labels={"Actual": "Actual delivery time", "Predicted": "Predicted delivery time"})
	scatter.add_shape(type="line", x0=actual.min().min(), y0=actual.min().min(), x1=actual.max().max(), y1=actual.max().max(), line=dict(color="#1f7a52", dash="dash"))
	scatter.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
	st.plotly_chart(scatter, use_container_width=True)
	st.info(f"MAE: {evaluation['mae']:.2f} minutes | R2 score: {evaluation['r2']:.2%} | Training rows: {len(data) - len(evaluation['actual']):,}")
