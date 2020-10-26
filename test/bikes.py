"""
Build forecasting models to predict bike share demand

@author: Nick
"""

# add the src directory to the system path
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

# dependencies
import json
import pandas as pd
from forecast import Forecasting

# import the configuration for modeling
with open("test/config.json") as f:
    config = json.load(f)

# produce a rolling forecast
model = Forecasting(**config)
pred, true = model.roll()
pred