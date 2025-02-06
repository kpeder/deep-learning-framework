from deeplearning.utils.data import split_data
from pandas import Series, DataFrame  # type: ignore

import logging


logger = logging.getLogger(__name__)


def test_split_data():
    '''
    Function to test the split_data function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''

    try:
        data = DataFrame(data={'val1': [1, 2, 3, 4], 'val2': [3, 4, 5, 6], 'sum': [4, 6, 8, 10]})
        assert type(data) is DataFrame
        X, A, y, z = split_data(dataframe=data, split=0.75, label_column='sum')
        assert isinstance(X, DataFrame) and len(list(X.columns)) == 2
        assert isinstance(y, Series) and len(y) == 3
        assert isinstance(A, DataFrame) and len(list(A.columns)) == 2
        assert isinstance(z, Series) and len(z) == 1
    except Exception as e:
        logger.exception(e)
        raise e
