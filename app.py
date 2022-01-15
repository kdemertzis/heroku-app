
"""
    Import library
"""
##-- PyCaret
import pycaret
from pycaret.classification import *
##-- Pandas
import pandas as pd
from pandas import Series, DataFrame
##-- explainerdashboard
import explainerdashboard

url = 'https://raw.githubusercontent.com/kdemertzis/Earthquakes/main/Data/Gradio/1_3class.csv'
# load the dataset
df = pd.read_csv(url)

# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)

rf = create_model('rf')

dashboard(rf)