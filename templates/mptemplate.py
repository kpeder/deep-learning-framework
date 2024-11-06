from contextlib2 import nullcontext
from deeplearning.utils.config import Config
from deeplearning.utils.logger import getContextLogger

import argparse
import datetime
import logging
import multiprocessing as mp
import os
import sys


'''
Set up the Python Logger using the configuration class defaults.
'''
handler: logging.Handler
logger: logging.Logger = logging.getLogger(__name__)

conf = Config()
conf.configure(config=None)  # load defaults
conf.configure(config={ "multiprocessing": { "enabled": True }})  # merge updates

try:
    formatter = logging.Formatter(conf.configuration["logging"]["format"])

    if conf.configuration["logging"]["type"] == 'stream':
        handler = logging.StreamHandler()
        handler.setStream(getattr(sys, conf.configuration["logging"]["path"]))

    if conf.configuration["logging"]["type"] == 'file':
        logdate = datetime.datetime.now()
        handler = logging.FileHandler(f'{os.environ["PWD"]}/log/{logdate.strftime("%Y%m%d")}_template.log')

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if hasattr(logging, conf.configuration["logging"]["level"].upper()):
        logger.setLevel(getattr(logging, conf.configuration["logging"]["level"].upper()))
        logger.warning(f'Loglevel has been set to {logger.getEffectiveLevel()} for log {__name__}.')

except Exception as e:
    raise e

logger.info(f'Loaded configuration: {conf.configuration}')


''' Configure argument parsing, for convenience.'''
parser = argparse.ArgumentParser()


''' Optional backend override.'''
parser.add_argument('--keras-backend-override', action='store', dest='keras_backend_override')

args = parser.parse_args()


''' Configure and import Keras.'''
os.environ["KERAS_BACKEND"] = (args.keras_backend_override or conf.configuration["keras"]["backend"])
logger.info(f'Configuring Keras backend as "{os.environ["KERAS_BACKEND"]}".')

import keras  # type: ignore # noqa: E402


logger.info(f'Using keras version {keras.__version__}.')


'''
Within the context of a logger to document the experiment,
do some experiment, log the activity and return the result.
'''
def do(logname: str=None, queue: mp.Queue=None):
    try:
        with getContextLogger(name=logname) as ctxtlogger:
            ctxtlogger.setLevel(logging.DEBUG)
            ctxtlogger.warning(f'Loglevel has been set to {ctxtlogger.getEffectiveLevel()} for log {logname}.')
            ctxtlogger.debug('A logline.')
            ctxtlogger.info(f'Process PID: {os.getpid()}')
            ctxtlogger.info(f'Parent PID: {os.getppid()}')
            ctxtlogger.debug('Another logline.')
            retval = logname, (os.getpid(), os.getppid())  # A convenient tuple that shows if we're multiprocessing
            if queue is not None:
                enqueue(retval, queue)
            return retval
    except Exception as e:
        raise e


''' A callback to log the result of some experiments.'''
def callback(results: list=None):
    try:
        for r in results:
            logger.info(r)
    except Exception as e:
        raise e


''' Dequeue wrapper.'''
def dequeue(queue: mp.Queue=None):
    try:
        if not queue.empty():
            result = do(*queue.get(block=False))
            return result
    except Exception as e:
        raise e


''' Enqueue wrapper.'''
def enqueue(item: tuple=None, queue: mp.Queue=None):
    try:
        queue.put(item)
        return
    except Exception as e:
        raise e


''' The worker pool with async map and callback.'''
with mp.Pool(conf.configuration["multiprocessing"]["workers"]) if conf.configuration["multiprocessing"]["enabled"] else nullcontext() as mpp:

    if mpp is not None:
        results = mpp.map_async(do,['__ctxt__','__iter__','__test__','__with__'], callback=callback)
        results.wait()
    else:
        results: list = []
        for name in ['__ctxt__','__iter__','__test__','__with__']:
            result = do(name)
            results.append(result)

        callback(results)


''' The parallel processing queue.'''
with mp.Manager() if conf.configuration["multiprocessing"]["enabled"] else nullcontext() as mpp:

    output = mpp.Queue()
    input = mpp.Queue()

    for args in [('__ctxt__', output),('__iter__', output),('__test__', output),('__with__', output)]:
        enqueue(args, input)

    proc_list: list=[]
    semaphore = mp.BoundedSemaphore(conf.configuration["multiprocessing"]["workers"])

    try:
        ''' While there's queued work, keep allocating processors.'''
        while not input.empty():
            if semaphore.acquire(block=False):
                proc = mp.Process(target=dequeue, args=(input,))
                proc.start()
                logger.info(f'Process {proc.name} with PID {proc.pid} acquired semaphore!')
                proc_list.append(proc)
            for proc in proc_list:
                if proc.exitcode is not None:
                    proc_list.remove(proc)
                    semaphore.release()
                    logger.info(f'Process {proc.name} with PID {proc.pid} released semaphore!')
    finally:
        ''' When the work is all allocated, wait for it to finish.'''
        while len(proc_list) > 0:
            for proc in proc_list:
                if proc.exitcode is not None:
                    proc_list.remove(proc)
                    semaphore.release()
                    logger.info(f'Process {proc.name} with PID {proc.pid} released semaphore!')

    try:
        results: list=[]
        while not output.empty():
            results.append(output.get())
        for result in results:
            logger.info(f'{result}')
    except Exception as e:
        raise e
