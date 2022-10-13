from flask import Flask, request
from utils.common import response_message
from utils.interpolation_methods import interpolation
from utils.outlier import timeseries_outlier , normal_outlier
from utils.balance import sampling
import khayyam as kh
import pandas as pd
from datetime import datetime
from persiantools.jdatetime import JalaliDate

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def isup():
    return response_message('API is active')


@app.route('/service1', methods=['GET', 'POST'])
def service1():
    req = request.get_json()
    data = pd.DataFrame(req['data'])
    config = pd.Series(req['config'])

    if config['type'] == 'miladi':
        data.time = pd.to_datetime(data.time, infer_datetime_format=True)

    elif config['type'] == 'shamsi':
        dates = []
        for d in data.time:
            tmp_d = kh.JalaliDatetime(*list(d.split('-'))).todatetime()
            tmp_d = pd.to_datetime(tmp_d)
            dates.append(tmp_d)
        data.time = dates

    result = interpolation(data, config)
    result = result.to_json()
    return response_message(dict({"data": result}))

@app.route('/service2', methods=['GET', 'POST'])
def service2():
    req = request.get_json()
    data = pd.DataFrame(req['data'])
    config = pd.Series(req['config'])
    data.time = pd.to_datetime(data.time)
    result = interpolation(data, config)
    for i in range(result['time'].size):
        result['time'].iloc[i] = datetime.timestamp(result['time'].iloc[i])
        result['time'].iloc[i] = JalaliDate.fromtimestamp(result['time'].iloc[i])
        result['time'].iloc[i] = str(result['time'].iloc[i])
    result = result.to_json()
    return response_message(dict({"data": result}))

@app.route('/service3', methods=['GET', 'POST'])
def service3():
    req = request.get_json()
    data = pd.DataFrame(req['data'])
    config = pd.Series(req['config'])
    feature = data.feature.copy()

    if config["time_series"]== True :
        out = timeseries_outlier(data , feature)

    elif config['time_series'] == False :
        data = data.set_index(data.id).drop('id', axis=1)
        out = normal_outlier(data)

    return response_message(out)

@app.route('/service4', methods=['GET', 'POST'])
def service4():
    req = request.get_json()
    data = pd.DataFrame(req['data'])
    config = pd.Series(req['config'])
    out = sampling(data, config)
    out = out.reset_index().to_json()
    out = {'data': out}
    return response_message(out)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)