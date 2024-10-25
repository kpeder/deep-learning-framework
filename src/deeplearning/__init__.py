import logging
import sys


logger = logging.getLogger(__name__)

try:
    formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
    handler = logging.StreamHandler()
    handler.setStream(sys.stdout)
    handler.setFormatter(formatter)
    logger.setLevel(logging.WARN)
    logger.addHandler(handler)
    logger.warning(f'Package loglevel has been set to {logger.getEffectiveLevel()}')
except Exception as e:
    logger.exception(e)
    raise e
