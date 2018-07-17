import pandas as pd
import quandl
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
# visualization
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt

#it so that we only show first 4 decimals for floats
# np.set_printoptions(precision=4, suppress=True)

#using files from local.

def get_stockdata(symbol):
    df = None
    if symbol == 'SPY':
        df = pd.read_csv("./stockdata/{0}".format(symbol)+".csv",
                         index_col=['Date'], usecols=['Date', 'Close', 'Volume'])
        df.columns = ['close', 'vol']
        return df


SPY_data = get_stockdata('SPY')
print(SPY_data.head())




#api call on quandl

df = quandl.get('WIKI/JPM')

print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print (df.head())
