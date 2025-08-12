from deeplearning.models.scikit_prediction import get_model
from deeplearning.utils.config import Config
from deeplearning.utils.data import split_data
from deeplearning.utils.filesystem import data_fetcher, get_data_from_bucket

import argparse
import datetime
import logging
import os
import pandas  # type: ignore
import sys


''' Set up the Python Logger using the configuration class defaults.'''
handler: logging.Handler
logger: logging.Logger = logging.getLogger(__name__)

conf = Config()
conf.configure(config=None)

try:
    formatter = logging.Formatter(conf.configuration["logging"]["format"])

    if conf.configuration["logging"]["type"] == 'stream':
        handler = logging.StreamHandler()
        handler.setStream(getattr(sys, conf.configuration["logging"]["path"]))

    if conf.configuration["logging"]["type"] == 'file':
        logdate = datetime.datetime.now()
        handler = logging.FileHandler(f'{os.environ["PWD"]}/log/{logdate.strftime("%Y%m%d")}_mnist.log')

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if hasattr(logging, conf.configuration["logging"]["level"].upper()):
        logger.setLevel(getattr(logging, conf.configuration["logging"]["level"].upper()))
        logger.warning(f'Loglevel has been set to {logger.getEffectiveLevel()} for log {__name__}.')

except Exception as e:
    raise e

''' Configure argument parsing, for convenience.'''
parser = argparse.ArgumentParser()

parser.add_argument('--bucket-name', dest='bucket_name', type=str)
parser.add_argument('--data-dir', dest='data_dir', type=str)
parser.add_argument('--data-split', dest='data_split', type=float, default=0.8)
parser.add_argument('--data-type', dest='data_type', type=str, default='csv')
parser.add_argument('--drop-columns', dest='drop_columns', type=str, default='')
parser.add_argument('--label-column', dest='label_column', type=str)
parser.add_argument('--model-dir', dest='model_dir', type=str)
parser.add_argument('--pipeline-name', dest='pipeline_name', type=str)
parser.add_argument('--source-data', dest='source_data', type=str)

args = parser.parse_args()

''' Compile columns to drop.'''
drop_columns = []
if args.drop_columns:
    drop_columns = args.drop_columns.split(',')

''' Get data and save to the pipeline bucket.'''
destination = 'gs://{}'.format('/'.join(list([args.bucket_name, args.data_dir, args.pipeline_name])))
location = data_fetcher(name=args.pipeline_name, source=args.source_data, dest=destination)
logger.info('Created pipeline data source at location {}'.format(location))

''' Load dataframe from pipeline bucket.'''
blobdir = '/'.join(list([args.data_dir, args.pipeline_name]))
data = get_data_from_bucket(bucket_name=args.bucket_name, data_dir=blobdir, data_type=args.data_type)
logger.info('Loaded raw dataframe with columns {}'.format(data.columns.values))

''' Data cleanup.'''
data.columns = [col.lower() for col in data.columns]
data.age = data.age.fillna(0).astype(int)  # we will re-fill these values with nearest neighbor values later
data.fare = data.fare.round(2)

''' One hot encoding for gender and embarkation point.'''
data.embarked = pandas.Categorical(data.embarked)
embarked_one_hot = pandas.get_dummies(data.embarked, prefix='embarked', dtype=int)
data.sex = pandas.Categorical(data.sex)
sex_one_hot = pandas.get_dummies(data.sex, prefix='gender', dtype=int)

cols = list(data.columns)

for index, col in enumerate(list(embarked_one_hot.columns)):
    data.insert(cols.index('embarked') + index, col, embarked_one_hot[col])

for index, col in enumerate(list(sex_one_hot.columns)):
    data.insert(cols.index('sex') + index, col, sex_one_hot[col])

data.drop(columns=['embarked', 'sex'], inplace=True)
data.rename(columns={'embarked_C': 'embarked_cherbourg',
                     'embarked_Q': 'embarked_queenstown',
                     'embarked_S': 'embarked_southampton'},
            inplace=True)

''' Convert string-y fields to NaN.'''
data.name = pandas.to_numeric(data.name, errors='coerce')
data.ticket = pandas.to_numeric(data.ticket, errors='coerce')
data.cabin = pandas.to_numeric(data.cabin, errors='coerce')

''' Implicitly drop columns containing NaN values. We're assuming that
sparse or arbitrary data such as passenger names, ticket numbers and cabin
numbers have either no impact or an adverse impact on classification due to
sparseness and unreliability of the data. We want passenger demographics such
as class, age, and family members aboard to be used, since they are more
informative metrics for generalized cases.'''
for col in list(data.columns):
    if data[col].isna().sum() > 0:
        data.drop(columns=[col], inplace=True)

logger.info('Cleaned up and encoded raw columns, new dataframe has columns {}.'.format(list(data.columns)))

''' Scale the remaining data with a division by its maximum value so that we
can use a nearest neighbor approach for filling in missing age data. We'll omit
the passengerid column, because we'll use this column as an identifier rather
than treating it as a feature.'''
scaled_data = pandas.DataFrame()
scaled_cols = []

for col in list(data.columns):
    if col not in ['age', 'passengerid']:
        scaled_data[col] = data[col] / data[col].max()
        scaled_cols.append(col)
    else:
        scaled_data[col] = data[col]

logger.info('Scaled data in columns {}.'.format(scaled_cols))

''' Use nearest neighbor regression model to fill in age preditions. The
n_neighbors value is set to minimize the average minkowski distance.'''
knn = get_model(model_type='knregressor', n_neighbors=2)

''' Drop records with missing age values, these are the prediction targets.'''
a_train, a_test, z_train, z_test = split_data(scaled_data.query('age != 0'),
                                              drop_columns=drop_columns,
                                              label_column='age',
                                              split=0.9)

a_features = list(a_train.columns)
logger.info('Using regression training features {}.'.format(a_features))

a_target = z_train.name
logger.info('Using regression target column \'{}\'.'.format(a_target))

''' Use arrays instead of dataframes for training data.'''
a_train = a_train.to_numpy()
a_test = a_test.to_numpy()
z_train = z_train.to_numpy()
z_test = z_test.to_numpy()
logger.info('Converted training data to ndarrays.')

knn.fit(a_train, z_train)
logger.info('Fit age regression model to average minkowski distance of {}.'.format(knn.score(a_test, z_test)))

''' Predict the age of every passenger.'''
age_predictions = pandas.Series()
age_predictions = round(scaled_data.drop(columns=['age', 'passengerid']).apply(lambda a: knn.predict([a])[0], axis=1)).astype(int)

''' Merge predicted ages in, only where age is 0 (unknown).'''
scaled_data['age'] = scaled_data['age'].combine(age_predictions, lambda d, s: s if d == 0 else d)
logger.info('Combined age predictions from regression model to fill in unknown ages.')

''' Scale the age column.'''
scaled_data['age'] = scaled_data['age'] / scaled_data['age'].max()
logger.info('Scaled data in column \'age\' to eliminate feature bias.')

''' Use random forest classifier model to predict passenger survival.'''
rfc = get_model('rfclassifier')

''' Split the dataset for classification model training and testing.'''
x_train, x_test, y_train, y_test = split_data(scaled_data,
                                              drop_columns=drop_columns,
                                              label_column=args.label_column,
                                              split=args.data_split)

features = list(x_train.columns)
label = y_train.name

x_features = list(x_train.columns)
logger.info('Using binary classification training features {}.'.format(x_features))

y_label = y_train.name
logger.info('Using binary classification label column \'{}\'.'.format(y_label))

''' Use arrays instead of dataframes for training data.'''
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
logger.info('Converted training data to ndarrays.')

rfc.fit(x_train, y_train)
logger.info('Fit survival classification model to average accuracy of {}.'.format(rfc.score(x_test, y_test)))
