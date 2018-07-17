import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


df = pd.read_csv("./stockdata/JPM.csv")
print(df.head())
df.set_index('Date', inplace=True)
df.to_csv('newJPM.csv')
df = pd.read_csv('newJPM.csv', index_col=0)
