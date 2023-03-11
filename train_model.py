from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def makeStationary(df):

    return df

def AugumentedDickeyFullerTest(train):
    adf = adfuller(train['point_value'])
    print("P value",adf[1])

    return adf

def preProcessingPipeline(df):
    df = df.dropna()
    df['date'] = pd.to_datetime(df['point_timestamp'], infer_datetime_format=True)
    df = df.set_index(['date'])

    return df

def dataPlot(df):
    plt.plot_date(df.index,df['point_value'])
    plt.show()

def graphPlot(pred,test):
    plt.plot_date(test.index,test['point_value'],color='cornflowerblue', label='Original')
    plt.plot_date(test.index,pred,color='firebrick', label='Predicted')
    plt.show()

def modelPipeline(train,test,df):
    model = ARIMA(train['point_value'], order=(2, 1, 2))
    ARIMA_model = model.fit()

    start = len(train)
    end = len(test) + len(train) - 1

    pred = ARIMA_model.predict(start=start, end=end)
    pred.index = df['point_timestamp'][start:end + 1]
    print("Predicted, Test Values ")
    print(pred,test)

    graphPlot(pred,test)

    mape = mean_absolute_percentage_error(df['point_value'][start:end + 1], pred)
    return mape

def model_train(df):
    df = preProcessingPipeline(df)
    train = df[0:len(df) // 2]
    test = df[len(df) // 2:]

    adf = AugumentedDickeyFullerTest(train)

    if adf[1]>0.05:
        print(" Nature of The TIME SERIES Data :  Non stationary")
        df = makeStationary(df)
        dataPlot(df)
        mape = modelPipeline(train,test,df)
    else:
        print("Nature of The TIME SERIES Data : Stationary")
        dataPlot(df)
        mape = modelPipeline(train,test,df)

    return mape