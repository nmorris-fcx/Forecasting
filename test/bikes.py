"""
Build forecasting models to predict bike share demand

@author: Nick
"""

# add the src directory to the system path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/') ))

import pandas as pd
from forecast import Forecasting
