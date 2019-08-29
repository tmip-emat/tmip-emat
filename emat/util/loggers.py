'''

This module contains code for logging emat processes.

'''

from functools import wraps
import inspect
import time
from contextlib import contextmanager

import logging
from logging import DEBUG, INFO

# Created on 23 dec. 2010
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['get_logger',
           'get_module_logger',
           'log_to_stderr',
           'DEBUG',
           'INFO',
           'DEFAULT_LEVEL',
           'LOGGER_NAME']
LOGGER_NAME = "EMAT"
DEFAULT_LEVEL = DEBUG
INFO = INFO


def create_module_logger(name=None):
    if name is None:
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        name = mod.__name__
    logger = logging.getLogger("{}.{}".format(LOGGER_NAME, name))

    _module_loggers[name] = logger
    return logger


def get_module_logger(name):
    try:
        logger = _module_loggers[name]
    except KeyError:
        logger = create_module_logger(name)

    return logger


_rootlogger = None
_module_loggers = {}
_logger = logging.getLogger(LOGGER_NAME)
_module_loggers[LOGGER_NAME] = _logger


def format_elapsed_time(duration_milliseconds):
    hours, rem = divmod(duration_milliseconds/1000, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    else:
        return ("{:0>2}:{:05.2f}".format(int(minutes),seconds))


class ElapsedTimeFormatter(logging.Formatter):
    def format(self, record):
        record.elapsedTime = format_elapsed_time(record.relativeCreated)
        return super(ElapsedTimeFormatter, self).format(record)

LOG_FORMAT = '[{elapsedTime}] {processName:s}/{levelname:s}: {message:s}'


def method_logger(name):
    logger = get_module_logger(name)
    classname = inspect.getouterframes(inspect.currentframe())[1][3]

    def real_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # hack, because log is applied to methods, we can get
            # object instance as first arguments in args
            logger.debug('calling {} on {}'.format(func.__name__, classname))
            res = func(*args, **kwargs)
            logger.debug('completed calling {} on {}'.format(func.__name__, classname))
            return res

        return wrapper

    return real_decorator


def get_logger():
    '''
    Returns root logger
    '''
    global _logger

    if not _logger:
        _logger = logging.getLogger(LOGGER_NAME)
        _logger.handlers = []
        _logger.addHandler(logging.NullHandler())
        _logger.setLevel(DEBUG)
        _module_loggers[LOGGER_NAME] = _logger

    return _logger


def log_to_stderr(level=None, top=False):
    '''
    Turn on logging and add a handler which prints to stderr

    Parameters
    ----------
    level : int
            minimum level of the messages that will be logged

    '''

    if not level:
        level = DEFAULT_LEVEL

    logger = get_logger() if not top else logging.getLogger()

    # avoid creation of multiple stream handlers for logging to console
    for entry in logger.handlers:
        if (isinstance(entry, logging.StreamHandler)) and \
                (entry.formatter._fmt == LOG_FORMAT):
            return logger

    formatter = ElapsedTimeFormatter(LOG_FORMAT, style='{')
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(level)

    return get_logger()

def timesize_stack(t):
    if t<60:
        return f"{t:.2f}s"
    elif t<3600:
        return f"{t//60:.0f}m {timesize_stack(t%60)}"
    elif t<86400:
        return f"{t//3600:.0f}h {timesize_stack(t%3600)}"
    else:
        return f"{t//86400:.0f}d {timesize_stack(t%86400)}"

@contextmanager
def timing_log(label=''):
    log = get_logger()
    start_time = time.time()
    log.critical(f"<TIME BEGINS> {label}")
    try:
        yield
    except:
        log.critical(f"<TIME ERROR!> {label} <{timesize_stack(time.time()-start_time)}>")
        raise
    else:
        log.critical(f"< TIME ENDS > {label} <{timesize_stack(time.time()-start_time)}>")
