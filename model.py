import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

train_cache = 'train_cache.csv'
test_cache = 'test_cache.csv'

num_epochs = 5
feature_columns = ['country', 'articleType', 'adPosition', 'isMobile', 'day_of_week', 'day_of_year', 'hour_of_day']

query = 'SELECT country, '
'articleType, '
'adPosition, '
'isMobile, '
'tsHour, '
'impressions, '
'clicks, '
'FROM <table> WHERE tsHour BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) AND CURRENT_DATE()'


def load_data(force=False):
    if force or not os.path.exists(train_cache) or not os.path.exists(test_cache):
        df = pd.read_gbq(query,
                         project_id='<project_id>',
                         private_key='<private_key>',
                         dialect='standard',
                         configuration={
                             'query': {
                                 'useQueryCache': False,
                                 'allowLargeResult': True
                             }
                         })
        df['tsHour'] = pd.to_datetime(df['tsHour'])
        df['day_of_week'] = df['tsHour'].dt.dayofweek
        df["day_of_year"] = df["tsHour"].dt.dayofyear
        df['hour_of_day'] = df["tsHour"].dt.hour
        df['ctr'] = df['clicks'] // df['impressions']
        train, test = train_test_split(df, test_size=.1)
        train.to_csv(train_cache)
        test.to_csv(test_cache)
    else:
        train = pd.read_csv(train_cache)
        test = pd.read_csv(test_cache)

    return train, test


def train_fn(df):
    feature_cols = predict_fn(df)
    label = tf.constant(df['ctr'].values)
    return feature_cols, label


def predict_fn(df):
    feature_cols = dict({k: tf.constant(df[k].values) for k in feature_columns})
    return feature_cols


def get_model():
    return tf.estimator.LinearRegressor(feature_columns=[tf.feature_column.categorical_column_with_hash_bucket('country', hash_bucket_size=100),
                                                         tf.feature_column.numeric_column('articleType'),
                                                         tf.feature_column.numeric_column('adPosition'),
                                                         tf.feature_column.numeric_column('isMobile'),
                                                         tf.feature_column.numeric_column('day_of_year'),
                                                         tf.feature_column.numeric_column('day_of_week'),
                                                         tf.feature_column.numeric_column('hour_of_day')],
                                        optimizer=tf.train.AdamOptimizer(),
                                        loss_reduction=tf.losses.Reduction.MEAN,
                                        model_dir='model')


train_data, test_data = load_data()
model = get_model()
model.train(input_fn=lambda: train_fn(train_data), steps=num_epochs)
summary = model.evaluate(input_fn=lambda: train_fn(test_data), steps=1)
print(summary)