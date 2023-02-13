# Snowpark for Python API reference: https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/index.html
# Snowpark for Python Developer Guide: https://docs.snowflake.com/en/developer-guide/snowpark/python/index.html

import calendar 
import altair as alt
import streamlit as st
import pandas as pd
import snowflake.snowpark as snowpark 
from snowflake.snowpark.functions import col

# Function to load last six months' budget allocations and ROI 
def load():
  data = session.table("BUDGET_ALLOCATIONS_AND_ROI").unpivot("Budget", "Channel", ["SearchEngine", "SocialMedia", "Video", "Email"]).filter(col("MONTH") != "July")
  alloc, rois, last_alloc = data.drop("ROI"), data.drop(["CHANNEL", "BUDGET"]).distinct(), data.filter(col("MONTH") == "June")
  return data.to_pandas(), alloc.to_pandas(), rois.to_pandas(), last_alloc.to_pandas()

def predict(budgets):
  pred = session.sql(f"SELECT predict_roi(array_construct({budgets[0]*1000},{budgets[1]*1000},{budgets[2]*1000},{budgets[3]*1000})) as PREDICTED_ROI").to_pandas()
  pred = pred["PREDICTED_ROI"].values[0] / 100000
  change = round(((pred / rois["ROI"].iloc[-1]) - 1) * 100, 1)
  return pred, change

def chart(chart_data):
  base = alt.Chart(chart_data).encode(alt.X("MONTH", sort=list(calendar.month_name), title=None))
  bars = base.mark_bar().encode(y=alt.Y("BUDGET", title="Budget", scale=alt.Scale(domain=[0, 300])), color=alt.Color("CHANNEL", legend=alt.Legend(orient="top", title=" ")), opacity=alt.condition(alt.datum.MONTH=="July", alt.value(1), alt.value(0.3)))
  lines = base.mark_line(size=3).encode(y=alt.Y("ROI", title="Revenue", scale=alt.Scale(domain=[0, 25])), color=alt.value("#808495"))
  points = base.mark_point(strokeWidth=3).encode(y=alt.Y("ROI"), stroke=alt.value("#808495"), fill=alt.value("white"), size=alt.condition(alt.datum.MONTH=="July", alt.value(300), alt.value(70)))
  chart = alt.layer(bars, lines + points).resolve_scale(y="independent").configure_view(strokeWidth=0).configure_axisY(domain=False).configure_axis(labelColor="#808495", tickColor="#e6eaf1", gridColor="#e6eaf1", domainColor="#e6eaf1", titleFontWeight=600, titlePadding=10, labelPadding=5, labelFontSize=14).configure_range(category=["#FFE08E", "#03C0F2", "#FFAAAB", "#995EFF"])
  st.altair_chart(chart, use_container_width=True)

# Streamlit config
st.header("SportsCo Ad Spend Optimizer")
st.subheader("Advertising budgets")

# Call functions to get Snowflake session and load data
session = snowpark.session._get_active_session()
channels = ["Search engine", "Email", "Social media", "Video"]
channels_upper = [channel.replace(" ", "").upper() for channel in channels]
data, alloc, rois, last_alloc = load()
last_alloc = last_alloc.replace(channels_upper, channels)

# Display advertising budget sliders and set their default values
col1, _, col2 = st.columns([4, 1, 4])
budgets = []
for alloc, col in zip(last_alloc.itertuples(), [col1, col1, col2, col2]):
  budgets.append(col.slider(alloc.CHANNEL, 0, 100, alloc.BUDGET, 5))

# Function to call "predict_roi" UDF that uses the pre-trained model for inference
# Note: Both the model training and UDF registration is done in Snowpark_For_Python.ipynb
pred, change = predict(budgets)
st.metric("", f"Predicted revenue ${pred:.2f} million", f"{change:.1f} % vs last month")
july = pd.DataFrame({"MONTH": ["July"]*4, "CHANNEL": channels_upper, "BUDGET": budgets, "ROI": [pred]*4})
chart(data.append(july).reset_index(drop=True).replace(channels_upper, channels))

# Setup the ability to save user-entered allocations and predicted value back to Snowflake 
if st.button("❄️ Save to Snowflake"):
  with st.spinner("Making snowflakes..."):
    df = pd.DataFrame({"MONTH": ["July"], "SEARCHENGINE": [budgets[0]], "SOCIALMEDIA": [budgets[1]], "VIDEO": [budgets[2]], "EMAIL": [budgets[3]], "ROI": [pred]})
    session.write_pandas(df, "BUDGET_ALLOCATIONS_AND_ROI")  
    st.success("✅ Successfully wrote budgets & prediction to your Snowflake account!")
    st.snow()
