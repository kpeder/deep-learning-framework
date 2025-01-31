from deeplearning.models.scikit_prediction import get_model
from deeplearning.utils.config import Config
from deeplearning.utils.data import split_data
from deeplearning.utils.filesystem import data_fetcher, get_data_from_bucket

import argparse
import datetime
import logging
import numpy  # type: ignore
import os
import pandas
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
print(blobdir)
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
sex_one_hot = pandas.get_dummies(data.sex, prefix='sex', dtype=int)

cols = list(data.columns.values)

for index, col in enumerate(list(embarked_one_hot.columns)):
    data.insert(cols.index('embarked') + index, col, embarked_one_hot[col])

for index, col in enumerate(list(sex_one_hot.columns)):
    data.insert(cols.index('sex') + index, col, sex_one_hot[col])

data.drop(columns=['embarked', 'sex'], inplace=True)

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
for col in list(data.columns.values):
    if data[col].isna().sum() > 0:
        data.drop(columns=[col], inplace=True)

print(data)

''' Scale the remaining data with a division by its maximum value so that we
can use a nearest neighbor approach for filling in missing age data. We'll omit
the passengerid column, because we'll use this column as an identifier rather
than treating it as a feature.'''
nn_data = pandas.DataFrame()
for col in list(data.columns.values):
    if col not in ['age', 'passengerid']:
        nn_data[col] = data[col] / data[col].max()
    else:
        nn_data[col] = data[col]

print(nn_data)

knn = get_model(model_type='knregressor')

''' Split the dataset for training and testing.'''
x_train, x_test, y_train, y_test = split_data(data, split=args.data_split, label_column=args.label_column, drop_columns=drop_columns)

features = list(x_train.columns)
label = y_train.name

logger.info('Split dataframe for training and testing, with label \'{}\' and features {}.'.format(label, features))

print(data.age.value_counts(normalize=True))
