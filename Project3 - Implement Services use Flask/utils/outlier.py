import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from prophet import Prophet
import pandas as pd

def timeseries_outlier(data, feature):

    #method1
    outliers1 = ['false' for i in range(len(feature))]
    m = Prophet()
    df = data.copy()
    df.time = pd.to_datetime(df.time)
    df.columns = ['ds', 'y']
    m.fit(df)
    test = pd.DataFrame(df['ds'])
    forecast = m.predict(test)
    for i in range(0, len(feature)):
        if(forecast['yhat_lower'].iloc[i] < df['y'].iloc[i] < forecast['yhat_upper'].iloc[i]):
            outliers1[i] = 'false'
        else : outliers1[i] = 'true'

    #method2
    data = data.set_index(data.time).drop('time', axis=1)
    model = AutoReg(feature, lags=len(feature) // 3)
    model_fit = model.fit()
    predictions = model_fit.predict(start=0, end=len(feature))
    print(predictions)
    outliers2 = ['false' for i in range(len(feature))]
    for i in range(len(feature)):
        if ((predictions[i+1] > feature[i]+feature[i]*2.5) or (predictions[i+1] < feature[i]-feature[i]*2.5)) :
            outliers2[i] = 'true'

    data['method1'] = outliers1
    data['method2'] = outliers2
    out = data.reset_index().to_json()
    out = {'data': out}
    return out


def normal_outlier(data):
    #method1
    outliers1 = []
    temp = ((data > data.mean() + 3 * data.std()) | (data < data.mean() - 3 * data.std()))
    for i in range (len(data)):
        if (temp['feature'].iloc[i]==False):
            outliers1.append('false')
        else : outliers1 = 'true'

    #method2
    outliers2 = []
    Q1 = data.quantile(0.4)
    Q3 = data.quantile(0.6)
    IQR = Q3 - Q1
    temp = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    for i in range(len(data)):
        if (temp['feature'].iloc[i] == False):
            outliers2.append('false')
        else:
            outliers2.append('true')

    data['method1'] = outliers1
    data['method2'] = outliers2
    out = data.reset_index().to_json()
    out = {'data': out}
    return out