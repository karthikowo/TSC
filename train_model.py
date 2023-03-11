from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

def graphPlot(pred):
    plt.style.use('seaborn')
    plt.plot_date(pred.index, pred, linestyle='solid')
    plt.show()

def modelPipeline(ts,test,df):
    model = ARIMA(ts['point_value'], order=(2, 1, 2))
    ARIMA_model = model.fit()
    print(ARIMA_model.summary())

    start = len(ts)
    end = len(test) + len(ts) - 1

    pred = ARIMA_model.predict(start=start, end=end)
    pred.index = df['point_timestamp'][start:end + 1]
    print(pred)

    graphPlot(pred)

    mape = mean_absolute_percentage_error(df['point_value'][start:end + 1], pred)
    return mape

def model_train(df):
    df = df.dropna()
    df['point_timestamp'] = pd.to_datetime(df['point_timestamp'])
    ts = df[0:len(df)//2]
    test = df[len(df)//2:]
    adf = adfuller(ts['point_value'])

    print("ADfuller stats")
    print(adf[0])
    print("P value",adf[1])
    print(adf[4])

    if adf[0]>adf[4]["5%"]:
        print("Non stationary")

        mape = modelPipeline(ts,test,df)
    else:
        print("Stationary")

        mape = modelPipeline(ts,test,df)

    return mape