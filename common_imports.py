import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold , cross_val_predict, learning_curve
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from xgboost import XGBRegressor 
from sklearn.svm import SVR
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings 
import torch
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
warnings.filterwarnings("ignore", category=FutureWarning)
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
plt.rc('font', family='AppleGothic')

sns.set_theme(style='whitegrid')  