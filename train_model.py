import sys

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')


def create_features(df):
    # extracting features from the dataset 
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.day_of_week
    df['month'] = df.index.month
    features = ['hour', 'dayofweek', 'month']
    df = df[features]
    return df


def arima_model(train, test, feature='point_value'):
    '''
    arima(p,d,q)
    the p value is calculated from the Partial Autocorrelation plot for which data points exceed the confidence interval.
    the q value is calculated from the Autocorrelation plot for which data points exceed the confidence interval.
    they are initially 0. as the ARIMA could be AR model,MA model,or the ARIMA model depending on the params.
    '''

    acf, cia = sm.tsa.acf(train[feature], alpha=0.05)
    pacf, cip = sm.tsa.pacf(train[feature], alpha=0.05)

    p = 0
    q = 0
    acf_absval = 0
    pacf_absval = 0

    for i in range(1, len(acf)):
        try:
            if abs(acf[i]) > cia[i][1] - cia[i][0]:
                if abs(acf[i]) > acf_absval:
                    p = i
                    acf_absval = abs(acf[i])

            if abs(pacf[i]) > cip[i][1] - cip[i][0]:
                if abs(pacf[i]) > pacf_absval:
                    q = i
                    pacf_absval = abs(pacf[i])
        except IndexError:
            continue

    model = ARIMA(train[feature], order=(p, 1, q))
    ARIMA_model = model.fit()

    start = len(train)
    end = len(test) + len(train) - 1

    pred = ARIMA_model.predict(start=start, end=end)
    pred.index = test['point_timestamp']

    return pred


def xgb_regressor(train, test, feature='point_value'):
    '''
    extracting features from the dataset and feeding into XGBoost algo
    '''
    X_train = create_features(train)
    Y_train = train[feature]
    X_test = create_features(test)
    Y_test = test[feature]

    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, Y_train)
    pred = reg.predict(X_test)
    return pred


def LSTMR(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train['point_value'].values.reshape(-1, 1))
    scaled_train = scaler.transform(train['point_value'].values.reshape(-1, 1))
    scaled_test = scaler.transform(test['point_value'].values.reshape(-1, 1))

    n_input = 10
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(generator)

    last_train_batch = scaled_train[-n_input:]
    last_train_batch = last_train_batch.reshape((1, n_input, n_features))

    test_predictions = []

    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    for i in range(len(test)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    true_preds = scaler.inverse_transform(test_predictions)
    return true_preds


def trainTestSplit(df):
    # splitting into train and test 
    train = df[0:int(len(df) * 0.5)]
    test = df[int(len(df) * 0.5):]
    return train, test


def makeStationary(df):
    # sample differencing 
    df['shift'] = df['point_value'].shift()
    df['diff'] = (df['point_value'] - df['shift']) / df['point_value'].rolling(window=12).mean()
    df = dropNullValues(df)
    df['point_value'] = df['diff']
    return df


def augmentedDickeyFullerTest(df, feature='point_value'):
    # adfuller test to check stationarity 
    adf = adfuller(df[feature])
    pval = adf[1]
    print("P value", pval)

    return pval


def dropNullValues(df):
    # drop null values in the dataframe 
    df = df.dropna()
    return df


def preProcessingPipeline(df, dateField, yvalField):
    '''
    the date,y values are set to a fixed name that will be used throughout.
    the index of the dataframe is made to date for functions that require this.
    '''
    df = dropNullValues(df)
    df['date'] = pd.to_datetime(df[dateField], infer_datetime_format=True)
    df['point_timestamp'] = df[dateField]
    df['point_value'] = df[yvalField]
    df = df.set_index(['date'])

    return df


def dataPlot(df):
    # plot the initial dataset 
    plt.plot_date(df.index, df['point_value'])
    plt.show()


def graphPlot(pred, test):
    # plotting test predictions vs original test set 
    plt.plot_date(test.index, test['point_value'], color='cornflowerblue', label='Original')
    plt.plot_date(test.index, pred, color='firebrick', label='Predicted')
    plt.show()


def modelClassifier(df, sdf):
    '''
    selecting the best model based on time series features.
    1. ADfuller test is performed in all the cases and the data is made stationary if needed.
    2. if the time series data has hour,day component the features are extracted and ensemble model is used. (XGBoost)
    3. if the data has seasonality we use the RNN model LSTM to capture the seasonality of the time series
    4. if the data has no daily,hourly component then we use the ARIMA model. which tunes the params(AR,MA,ARIMA)
    '''

    pval = augmentedDickeyFullerTest(df)
    fdf = create_features(df)
    modelused = ""
    print("BEFORE:", pval, sdf['trend'].mean(), sdf['seasonality'].mean())

    if fdf['hour'].sum() > 0 or fdf['dayofweek'].sum() > 0:
        # choose ensemble method or neural nets
        if pval < 0.05:
            print(f"Stationary Time Series data. pvalue = {pval}")
            train, test = trainTestSplit(df)
            if sdf['seasonality'].mean() > 15:
                pred = LSTMR(train, test)
                modelused = "LSTM"
            else:
                pred = xgb_regressor(train, test)
                modelused = "XGBoost"
        else:
            print(f"Non Stationary Time Series data. pvalue = {pval}")
            df = makeStationary(df)
            pval = augmentedDickeyFullerTest(df)
            print(f"Eliminated Trend Component. pvalue = {pval}")
            train, test = trainTestSplit(df)
            sdf = summary(df)

            if sdf['seasonality'].mean() > 15:
                pred = LSTMR(train, test)
                modelused = "LSTM"
            else:
                pred = xgb_regressor(train, test)
                modelused = "XGBoost"
    else:
        if pval < 0.05:
            print(f"Stationary Time Series data. pvalue = {pval}")

            train, test = trainTestSplit(df)
            if sdf['seasonality'].mean() > 15:
                pred = LSTMR(train, test)
                modelused = "LSTM"
            else:
                pred = arima_model(train, test)
                modelused = "ARIMA"
        else:
            print(f"Non Stationary Time Series data. pvalue = {pval}")
            df = makeStationary(df)
            pval = augmentedDickeyFullerTest(df)
            print(f"Eliminated Trend Component. pvalue = {pval}")
            train, test = trainTestSplit(df)
            sdf = summary(df)

            if sdf['seasonality'].mean() > 15:
                pred = LSTMR(train, test)
                modelused = "LSTM"
            else:
                pred = arima_model(train, test)
                modelused = "ARIMA"

    print("AFTER", pval, sdf['trend'].mean(), sdf['seasonality'].mean())

    mape = mean_absolute_percentage_error(test['point_value'], pred)
    return pred, test, modelused, mape


def modelPipeline(df):
    # feeding the dataset to model classifier 
    sdf = summary(df)
    pred, test, model, mape = modelClassifier(df, sdf)
    graphPlot(pred, test)

    return mape, model


def summaryPlot(df, sdf):
    # plotting the time series features 
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(sdf, label='Original time series', color='blue')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(df['trend'], label='Trend of time series', color='blue')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(df['seasonality'], label='Seasonality of time series', color='blue')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(df['resid'], label='Decomposition residuals of time series', color='blue')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def summary(df):
    # base summary
    print(df.describe())
    print("Kurtosis", df['point_value'].kurt())
    print("Skewness", df['point_value'].skew())

    # capturing trend,seasonality
    sdf = df[['point_timestamp', 'point_value']]
    sdf = sdf.set_index(['point_timestamp'])

    add_results = seasonal_decompose(sdf, model='additive', period=1)

    new_df_add = pd.concat([add_results.seasonal, add_results.trend, add_results.resid, add_results.observed], axis=1)
    new_df_add.columns = ['seasonality', 'trend', 'resid', 'observed']

    return new_df_add


def model_train(df, dateField, yvalField):
    # main function
    df = preProcessingPipeline(df, dateField, yvalField)
    dataPlot(df)
    mape, model = modelPipeline(df)

    return mape, model
