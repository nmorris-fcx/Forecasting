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

# add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

from forecast import Forecasting
from lasso import Regression