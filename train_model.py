from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def createFeatures(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.day_of_week
    df['month'] = df.index.month
    FEATURES = ['hour','dayofweek','month']
    df = df[FEATURES]
    return df

def arimaModel(train,test,feature='point_value'):
    model = ARIMA(train[feature], order=(2, 1, 2))
    ARIMA_model = model.fit()

    start = len(train)
    end = len(test) + len(train) - 1

    pred = ARIMA_model.predict(start=start, end=end)
    pred.index = test['point_timestamp']

    return pred

def xgbRegressor(train,test,feature='point_value'):
    X_train = createFeatures(train)
    Y_train = train[feature]
    X_test = createFeatures(test)
    Y_test = test[feature]

    print("XGB")
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, Y_train)
    pred = reg.predict(X_test)
    return pred

# def LSTM(train,test):


def trainTestSplit(df):
    train = df[0:int(len(df) * 0.75)]
    test = df[int(len(df) * 0.75):]
    return train,test

def makeStationary(df):

    df['shift'] = df['point_value'].shift()
    df['diff'] = (df['point_value'] - df['shift'])/df['point_value'].rolling(window=12).mean()
    df = dropNullValues(df)
    df['point_value'] = df['diff']
    return df

def AugumentedDickeyFullerTest(df,feature='point_value'):
    adf = adfuller(df[feature])
    pval = adf[1]
    print("P value",pval)

    return pval

def dropNullValues(df):
    df = df.dropna()
    return df

def preProcessingPipeline(df,dateField,yvalField):
    df = dropNullValues(df)
    df['date'] = pd.to_datetime(df[dateField], infer_datetime_format=True)
    df['point_timestamp'] = df[dateField]
    df['point_value'] = df[yvalField]
    df = df.set_index(['date'])

    return df

def dataPlot(df):
    plt.plot_date(df.index,df['point_value'])
    plt.show()

def graphPlot(pred,test):
    plt.plot_date(test.index,test['point_value'],color='cornflowerblue', label='Original')
    plt.plot_date(test.index,pred,color='firebrick', label='Predicted')
    plt.show()

def modelClassifier(train,test):

    pred1 = arimaModel(train,test)
    pred2 = xgbRegressor(train,test)

    mape = mean_absolute_percentage_error(test['point_value'], pred2)
    return pred2,"xgb",mape

def modelPipeline(df):

    train, test = trainTestSplit(df)
    pred,model,mape = modelClassifier(train,test)
    graphPlot(pred,test)

    return mape

def summary(df):
    print(df.describe())
    print("Kurtosis",df['point_value'].kurt())
    print("Skewness",df['point_value'].skew())

def model_train(df,dateField,yvalField):
    df = preProcessingPipeline(df,dateField,yvalField)
    summary(df)
    pval = AugumentedDickeyFullerTest(df)

    if pval>0.05:
        print("Nature of The TIME SERIES Data :  Non stationary")
        dataPlot(df)
        df = makeStationary(df)
        dataPlot(df)

        mape = modelPipeline(df)
    else:
        print("Nature of The TIME SERIES Data : Stationary")
        dataPlot(df)
        mape = modelPipeline(df)

    return mape