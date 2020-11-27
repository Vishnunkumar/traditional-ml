from statsmodels.tsa.api import ExponentialSmoothing seasonal_decompose
import statsmodels
import pandas as pd
import numpy as np
import datetime
from sklearn import metrics

def preprocess(df):
    """
    Preprocess the dataframe to required timeseries format
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df.fillna(method='ffill', inplace=True)
    
    for i in range(0, len(df)):
        df['Page.Loads'].iloc[i] = int(df['Page.Loads'].iloc[i].replace(',',''))
        df['Unique.Visits'].iloc[i] = int(df['Unique.Visits'].iloc[i].replace(',',''))
        df['First.Time.Visits'].iloc[i] = int(df['First.Time.Visits'].iloc[i].replace(',',''))
        df['Returning.Visits'].iloc[i] = int(df['Returning.Visits'].iloc[i].replace(',',''))

    return df


def rolling_forecast_es(df, df_test, p):
    """
    Does rolling training and forecast for one week at a time 
    """
    
    df_test['preds'] = 0
    for k in range(0, int(len(df_test)/p)):
        model = ExponentialSmoothing(np.asarray(df['Page.Loads'].iloc[:ix2 + (k)*p]),seasonal_periods=365, seasonal='add', trend='add')
        model_fit = model.fit()
        preds = model_fit.forecast(p)
        print(str(k) + 'th week')

        print("train-start-date:", df['Date'].iloc[0])
        print("train-end-date:", df['Date'].iloc[ix2+(k)*p])

        print('forecast-start-date:', df_test['Date'].iloc[(k)*p])
        print('forecast-end-date:', df_test['Date'].iloc[(k+1)*p])

        df_test['preds'].iloc[(k)*p:(k+1)*p] = preds
    
    return df_test


df = pd.read_csv('web-traffic.csv')
df = preprocess(df)

# Splitting the df to train and test
start_training = datetime.datetime(2015, 1, 1)
end_training = datetime.datetime(2019, 12, 30)

ix1 = df[df['Date'] == start_training].index[0]
ix2 = df[df['Date'] == end_training].index[0]

df_un = df[['Date', 'Page.Loads']]
df_train = df_un.iloc[ix1:ix2,:]
df_test = df_un.iloc[ix2:,:]

# forecast for the next month
df_om_es = rolling_forecast_es(df, df_test, 30)

# forecast for the next week
df_ow_es = rolling_forecast_es(df, df_test, 7)

# forecast for the next day
df_od_es = rolling_forecast_es(df, df_test, 1)
