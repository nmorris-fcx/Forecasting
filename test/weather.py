# -*- coding: utf-8 -*-
"""
Build forecasting models to predict temperature

@author: Nick
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

# add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

from forecast import Forecasting
from lasso import Regression

# In[1]: Prepare the data for modeling
data = (
    pd.read_csv("test/weather.csv").iloc[:, 1:].drop(columns=["year", "month", "day"])
)

# replace T in precip with a small number (T stands for trace amount)
data.loc[data["precip"] == "T", "precip"] = 0.001
data["precip"] = data["precip"].astype(float)

# replace NaN in events with "none"
data.loc[pd.isna(data["events"]), "events"] = "none"

# convert events into frequencies for each event category
text = data[["events"]].astype(str)
matrix = pd.DataFrame()
for c in text.columns:
    vector = TfidfVectorizer()
    matrix2 = vector.fit_transform(text[c].tolist())  # collect the unique words and compute their inverse frequencies for each sample
    matrix2 = pd.DataFrame(matrix2.toarray(), columns=vector.get_feature_names())
    matrix2.columns = [f"{c}_{i}" for i in matrix2.columns]
    matrix = pd.concat([matrix, matrix2], axis="columns")

# add frequency features to the data
data = data.drop(columns=["events"])
data = pd.concat([data, matrix], axis="columns")

# fill in missing values with linear interpolation
# then backfill incase the first row contains missing values
Auckland = (
    data.loc[data["city"] == "Auckland"].copy().interpolate().fillna(method="bfill")
)
Beijing = (
    data.loc[data["city"] == "Beijing"].copy().interpolate().fillna(method="bfill")
)
Chicago = (
    data.loc[data["city"] == "Chicago"].copy().interpolate().fillna(method="bfill")
)
Mumbai = data.loc[data["city"] == "Mumbai"].copy().interpolate().fillna(method="bfill")
SanDiego = (
    data.loc[data["city"] == "San Diego"].copy().interpolate().fillna(method="bfill")
)
data = pd.concat([Auckland, Beijing, Chicago, Mumbai, SanDiego], axis="index")

# choose a city to model
city = "Chicago"  # Auckland, Beijing, Chicago, Mumbai, San Diego
data.loc[data["city"] == city].to_csv("test/weather_v2.csv", index=False)

# import the configuration for modeling
with open("test/weather.json") as f:
    config = json.load(f)

# update the configuration for the new version of the data
config["csv"] = "test/weather_v2.csv"
config["inputs"] = data.drop(columns=["city", "date", "avg_temp"]).columns.tolist()

# test features
# config["inputs"] = None
# config["resolution"] = None
config["input_history"] = True

# In[2]: Model the data

# produce a rolling forecast
model = Regression(**config)
model.roll(verbose=True)

# compare model with baseline (exponential smoothing)
baseline_model = Forecasting(**config)
baseline_model.roll(verbose=True)

print(f"Lasso Average Error: {np.round(model._error.mean()[0] * 100, 2)}%")
print(f"Baseline Average Error: {np.round(baseline_model._error.mean()[0] * 100, 2)}%")

# In[3]: Analyze the model

# pick a step ahead to evaluate
step_ahead = 1
df = pd.concat(
    [model._actual.iloc[:, step_ahead - 1], model._predictions.iloc[:, step_ahead - 1]],
    axis="columns",
)
df.columns = ["Actual", "Predict"]
df["index"] = pd.to_datetime(df.index)

# report R2
print(
    f'step ahead={step_ahead}, R2={np.round(r2_score(df["Actual"], df["Predict"]) * 100, 2)}%'
)

# plot the prediction series
fig = px.scatter(df, x="index", y="Predict")
fig.add_trace(
    go.Scatter(
        x=df["index"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"
    )
)
fig.update_layout(font=dict(size=16))
fig.show()
