from contextlib2 import nullcontext
from deeplearning.utils.config import Config
from deeplearning.utils.process import callback_logger, dequeue, enqueue, do

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
conf.configure(config={"multiprocessing": {"enabled": True}})  # merge updates

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


''' Use a worker pool with async map and callback.'''
with mp.Pool(conf.configuration["multiprocessing"]["workers"]) if conf.configuration["multiprocessing"]["enabled"] else nullcontext() as mpp:

    if mpp is not None:
        results = mpp.map_async(do, ['__ctxt__', '__iter__', '__test__', '__with__'], callback=callback_logger)
        results.wait()
    else:
        result_list: list = []
        for name in ['__ctxt__', '__iter__', '__test__', '__with__']:
            result = do(name)
            result_list.append(result)

        callback_logger(result_list, __name__)


''' Perform threaded parallel processing from a work queue.'''
with mp.Manager() if conf.configuration["multiprocessing"]["enabled"] else nullcontext() as mpp:

    if mpp is not None:
        ''' In and out queues.'''
        inpipe = mpp.Queue()
        outpipe = mpp.Queue()

        ''' Items to process in a list of tuples.'''
        args_list: list = [('__ctxt__', outpipe), ('__iter__', outpipe), ('__test__', outpipe), ('__with__', outpipe)]
        for args in args_list:
            enqueue(args, inpipe)  # type: ignore[arg-type]

        ''' Primitives for worker coordination.'''
        proc_list: list = []
        semaphore = mp.BoundedSemaphore(conf.configuration["multiprocessing"]["workers"])

        try:
            ''' While there's queued work, keep allocating processors to free threads.'''
            while not inpipe.empty():
                ''' Get a free thread, configure and start a process, and add it to a list for tracking.'''
                if semaphore.acquire(block=False):
                    args = dequeue(inpipe)
                    proc = mp.Process(target=do, args=args)
                    proc.start()
                    logger.info(f'Started process {proc.name} with PID {proc.pid}.')
                    proc_list.append(proc)

                ''' Prune the process list and release threads as work completes.'''
                for proc in proc_list:
                    if proc.exitcode is not None:
                        proc_list.remove(proc)
                        semaphore.release()
                        logger.info(f'Finished process {proc.name} with PID {proc.pid}.')

        finally:
            ''' When the work is all allocated, wait for it to finish.'''
            while len(proc_list) > 0:
                for proc in proc_list:
                    if proc.exitcode is not None:
                        proc_list.remove(proc)
                        semaphore.release()
                        logger.info(f'Finished process {proc.name} with PID {proc.pid}.')

        try:
            ''' Process the results from the output queue.'''
            result_list = []
            while not outpipe.empty():
                result_list.append(dequeue(outpipe))

        finally:
            callback_logger(result_list, __name__)

    else:
        result_list = []
        for name in ['__ctxt__', '__iter__', '__test__', '__with__']:
            result = do(name)
            result_list.append(result)

        callback_logger(result_list, __name__)
