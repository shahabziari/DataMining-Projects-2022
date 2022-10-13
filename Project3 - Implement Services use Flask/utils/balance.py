import pandas as pd
from imblearn.over_sampling import SMOTE , RandomOverSampler
from imblearn.under_sampling import TomekLinks , ClusterCentroids  , RandomUnderSampler

def sampling(data, config):

    if config.method == 'SMOTE':
        min_num = data['class'].value_counts().values[-1]
        if min_num <= 1:
            data = pd.concat([data, data]).reset_index().drop('index', axis=1)
        oversample = SMOTE(k_neighbors = min(min_num, 6))
        x, y = oversample.fit_resample(data.drop(['class', 'id'], axis=1), data['class'])

    elif config.method == 'Oversampling':
        model = RandomOverSampler(random_state=1)
        x, y = model.fit_resample(data.drop(['class', 'id'], axis=1), data['class'])

    elif config.method == 'Tomeklinks':
        model = TomekLinks(sampling_strategy = 'auto')
        x, y = model.fit_resample(data.drop(['class', 'id'], axis=1), data['class'])

    elif config.method == 'Clustercentroids':
        model = ClusterCentroids()
        x, y = model.fit_resample(data.drop(['class', 'id'], axis=1), data['class'])

    elif config.method == 'UnderSampling':
        model = RandomUnderSampler(random_state=1)
        x, y = model.fit_resample(data.drop(['class', 'id'], axis=1), data['class'])

    x['class'] = y
    x['id'] = x.index.to_numpy() + 1

    out = pd.DataFrame(x, columns=data.columns)
    return out