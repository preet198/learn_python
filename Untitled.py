
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler

# visualization
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt

#make it so that we only show first 4 decimals for floats
# np.set_printoptions(precision=4,suppress=True)


# In[24]:


# read data into a DataFrame
def get_stockdata(symbol):
    df = None
    if symbol == 'SPY':
        # pd.read_csv reads csv file data
        df = pd.read_csv("./stockdata/{0}".format(symbol)+".csv", index_col=['Date'], usecols=['Date','Close','Volume'])
        df.columns = ['close','vol']
        return df
    if symbol == 'VIX':
        df = pd.read_csv("./stockdata/{0}".format(symbol)+".csv", index_col=['Date'], usecols=['Date','Close'])
        df.columns = ['VIX_close']
        return df
    if symbol == 'SPX_PC':
        df = pd.read_csv("./stockdata/{0}".format(symbol)+".csv", index_col=['DATE'], usecols=['DATE','P/C Ratio'])
        df.columns = ['put_call_ratio']
        return df
    if symbol not in ['SPY','VIX']:
        df = pd.read_csv("./stockdata/{0}".format(symbol)+".csv", index_col=['Date'],usecols=['Date','Adj Close','Volume'])
        df.columns = ['close','vol']
        return df


# In[25]:


# Read csv data in dataframe
stock_data = get_stockdata('JPM') # Predict future price of this stock 'JPM'
print (stock_data.tail())
# stock_data = stock_data.sort_index()
SPY_data = get_stockdata('SPY') # S&P 500 - gives direction of overall market
print (SPY_data.head())
# SPY_data = SPY_data.sort_index()
VIX_data = get_stockdata('VIX') # Volatility index of S&P 500
print (VIX_data.head())
# VIX_data = VIX_data.sort_index()
Put_Call_data = get_stockdata('SPX_PC') # put call ratio of S&P 500
print (Put_Call_data.tail())
# Put_Call_data = Put_Call_data.sort_index()


# In[26]:


stock_data.index = stock_data.index.map(lambda x: pd.to_datetime(x))
SPY_data.index = SPY_data.index.map(lambda x: pd.to_datetime(x))
VIX_data.index = VIX_data.index.map(lambda x: pd.to_datetime(x))
Put_Call_data.index = Put_Call_data.index.map(lambda x: pd.to_datetime(x))


# In[27]:


# Merge JPY, SPY, VIX and Put_call data
stock_data = stock_data.join(SPY_data, how="inner", rsuffix="_SPY")
stock_data = stock_data.join(VIX_data, how="inner")
stock_data = stock_data.join(Put_Call_data, how="inner")

stock_data = stock_data.sort_index(ascending=True)


# In[28]:


print (stock_data.index)


# In[29]:


# Prediction of future stock price on n'th day
days_to_look_forward = 5


# In[30]:


# Target variable 'y' is the stock price after certain days = n days
stock_data["future_date"]=pd.datetime(1990,1,1) # Initialize
print(stock_data.tail(10))
stock_data["future_date"][:-days_to_look_forward] = stock_data.index[days_to_look_forward:]
print(stock_data.tail(10))
stock_data["future_price"]=-1 # Initialize
print(stock_data.tail(10))
stock_data["future_price"][:-days_to_look_forward] = stock_data["close"][days_to_look_forward:]
print(stock_data.tail(10))
#note: date of 1900-01-01 and future price of -1 signifies the lack of sufficient prod data


# In[31]:


stock_data[-days_to_look_forward-1:].tail()


# In[32]:


stock_data["daily_return"]=0.000000001
stock_data["daily_return"][1:]=np.log(stock_data["close"][1:]/stock_data["close"][:-1].values)


# In[33]:


stock_data["daily_return"].tail()


# In[34]:


# gets rolling mean for 5 day window.
stock_data["rolling_mean"]=stock_data["close"].rolling(window=days_to_look_forward).mean()
stock_data["rolling_sd"]=stock_data["close"].rolling(window=days_to_look_forward).std()
stock_data["rolling_mean"].fillna(0,inplace=True)
stock_data["rolling_sd"].fillna(0,inplace=True)


# In[35]:


stock_data.tail()


# In[36]:


def get_bollinger_bands(x):
    upper_band, lower_band = x["rolling_mean"] + 2*x["rolling_sd"], x["rolling_mean"] - 2*x["rolling_sd"]
    return upper_band, lower_band


# In[37]:


stock_data["bollinger_band"]=stock_data[["rolling_mean","rolling_sd"]].apply(get_bollinger_bands,axis=1)
stock_data["upper_bollinger_band"]=stock_data["bollinger_band"].apply(lambda x: x[0])
stock_data["lower_bollinger_band"]=stock_data["bollinger_band"].apply(lambda x: x[1])
stock_data = stock_data.drop("bollinger_band", axis = 1)


# In[38]:


stock_data.fillna(method="bfill",inplace=True)


# In[39]:


stock_data.head()


# In[77]:


plt. plot(stock_data["upper_bollinger_band"],stock_data["lower_bollinger_band"])


# In[41]:


stock_data = stock_data[stock_data.columns.tolist()[0:6]+stock_data.columns.tolist()[8:]+stock_data.columns.tolist()[6:8]]
stock_data.columns.tolist()


# In[42]:


stock_data[-10:].head(10)


# In[43]:


features_for_scaling = stock_data.columns.tolist()[:-2]
print (features_for_scaling)


# In[44]:


scaler = StandardScaler()
X = stock_data[features_for_scaling]
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,columns=["scaled_"+i for i in features_for_scaling])


# In[50]:


stock_data.iloc[:23]
stock_data.head()


# In[51]:


# Vertical merge the scaled data with original
stock_data = stock_data.reset_index()   #To retain the index , which corresponds to Date column
stock_data=pd.concat([X_scaled,stock_data],axis=1)


# In[52]:


stock_data.index


# In[53]:


# set index to zero beacuse concat brought in two dates and number @ index[0]
stock_data = stock_data.set_index('Date')


# In[56]:


print (stock_data.index.max(), stock_data.index.min())


# In[58]:


# print(classification_report(y_test, y_pred))
# making copy of stock_data & getting date out of middle of table with days to look forwad metric.
# saving data in master_stock_data. at this point will not modify stock_data.
master_stock_data = stock_data.copy()
columns_replaced_with_underscores= [np.str.replace(i," ","_") for i in master_stock_data.columns.tolist()]
master_stock_data = master_stock_data[columns_replaced_with_underscores]
master_stock_data = master_stock_data[days_to_look_forward - 1:-days_to_look_forward]


# In[59]:


stock_data[16:].head()


# In[60]:


master_stock_data.head()


# In[61]:


# Split entire data in test data and prod data
prod_data_window = 90

# Strip test_train data and prod data according to "days_to_look_forward"
prod_data = master_stock_data[-prod_data_window:]
test_train_data = master_stock_data[:-prod_data_window]


# In[64]:


prod_data.shape
test_train_data.shape


# In[66]:


X_features = [i for i in test_train_data.columns.tolist() if 'scaled' in i]
print (X_features)


# In[67]:


X = test_train_data[X_features]
y = test_train_data.future_price
prod_X = prod_data[X_features]


# In[68]:


X_mult_train, X_mult_test, y_mult_train, y_mult_test = train_test_split(X, y, test_size=0.1, random_state=234)
print ("training data size:",X_mult_train.shape)
print ("testing data size:",X_mult_test.shape)
#train on training set
mult_linreg1 = LinearRegression()
mult_linreg1.fit(X_mult_train, y_mult_train)

#generate predictions on test set and evaluate
y_mult_pred_test = mult_linreg1.predict(X_mult_test)
print ("Prediction set RMSE:",np.sqrt(np.abs(metrics.mean_squared_error(y_mult_test, y_mult_pred_test))))


# In[70]:


prod_data_preds = mult_linreg1.predict(prod_X)
print ("prod_data_preds - RMSE:",np.sqrt(np.abs(mean_squared_error(prod_data.future_price,prod_data_preds))))

# new column predictor to prod data (with least RMSE)
prod_data["predicted_price"]=prod_data_preds


# In[72]:


# Plotting graph with x="future_date", and y = ["future_price","prediction_close"]
df = prod_data[['future_date','future_price','predicted_price']]
df = df.reset_index()
df.head()


# In[73]:


plt.plot(master_stock_data['close'])

fig, ax = plt.subplots(figsize=(11, 8))
ax.plot(prod_data.index, prod_data.future_price, label='Future_price_5_days')
ax.plot(prod_data.index, prod_data.predicted_price, label='Predicted_price')
ax.set_ylabel("JPM Stock price($)")
ax.legend()
