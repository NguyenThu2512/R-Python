#%%-import library
import time
import warnings
import numpy as np
import json
import seaborn as sns
from pandas.io.json import json_normalize
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sms
import ijson
from pandas.io.json import json_normalize
from seaborn import heatmap
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')
import xml.etree.ElementTree as ET
import pandas as pd
#%%- load data
df=pd.read_excel("./Data/VNM_2016_2020", index_col=0)
print(df)
