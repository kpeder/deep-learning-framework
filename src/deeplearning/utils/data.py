from sklearn.model_selection import train_test_split

import logging
import pandas


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
logger.warning('Loglevel for module {} set to {}'.format(__name__, logger.getEffectiveLevel()))


def split_data(dataframe: pandas.DataFrame, split: float, label_column: str, drop_columns: list = []):
    ''' Split a dataframe into train and test feature data with train and test labels.'''

    features = list(dataframe.columns)

    try:
        ''' Pop the label column off of the feature list.'''
        features.pop(features.index(label_column))
    except Exception as e:
        logger.warning('Invalid label column \'{}\'.'.format(label_column))

    try:
        ''' Pop the drop columns off of the feature list.'''
        for column in drop_columns:
            features.pop(features.index(column))
            logger.info('Dropped feature column \'{}\'.'.format(column))
    except Exception as e:
        logger.warning('Invalid drop column \'{}\'.'.format(column))

    ''' Split the feature data and labels.'''
    x = dataframe[features]
    y = dataframe[label_column]

    ''' Create the training data with configured splits.'''
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split, random_state=72)

    ''' Training data, test data, training labels, test labels.'''
    return x_train, x_test, y_train, y_test
