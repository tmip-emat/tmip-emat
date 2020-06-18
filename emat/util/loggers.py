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


def log_to_stderr(level=None, top=False, workbench=True):
    '''
    Turn on logging and add a handler which prints to stderr

    Parameters
    ----------
    level : int
            minimum level of the messages that will be logged

    '''

    if not level:
        level = DEFAULT_LEVEL

    if workbench:
        from ..workbench.util import ema_logging
        ema_logging.LOGGER_NAME = LOGGER_NAME

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

    if workbench:
        from ..workbench.util import ema_logging
        ema_logging._rootlogger = logger
        import importlib
        existing_module_loggers = list(ema_logging._module_loggers.keys())
        ema_logging._module_loggers.clear()
        for module_name in existing_module_loggers:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                pass
            else:
                module._logger = ema_logging.get_module_logger(module_name)

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


class TimingLog:

    def __init__(self, label='', log=None, level=50):
        if log is None:
            log = get_logger()
        self.label = label
        self.log = log
        self.level = level
        self.split_time = None
        self.current_task = ''

    def __enter__(self):
        self.start_time = time.time()
        self.log.log(self.level, f"<BEGIN> {self.label}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        now = time.time()
        if self.split_time is not None:
            self.log.log(self.level, f"<SPLIT> {self.label} <{timesize_stack(now - self.split_time)}>")
        if exc_type is None:
            self.log.log(self.level, f"<-END-> {self.label} <{timesize_stack(now - self.start_time)}>")
        else:
            self.log.log(self.level, f"<ERROR> {self.label} <{timesize_stack(now - self.start_time)}>")

    def split(self, note=''):
        if self.split_time is None:
            self.split_time = self.start_time
        now = time.time()
        if note:
            note = " / " + note
        self.log.log(self.level, f"<SPLIT> {self.label}{note} <{timesize_stack(now - self.split_time)}>")
        self.split_time = now



try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

class OutputWidgetHandler(logging.Handler):
    """ Custom logging handler sending logs to an output widget """

    def __init__(self, *args, **kwargs):
        if widgets is None:
            raise ModuleNotFoundError('ipywidgets')
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            'width': '100%',
            'height': '160px',
            'border': '1px solid black',
            'overflow': 'scroll',
        }
        self.out = widgets.Output(layout=layout)

    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record+'\n'
        }
        self.out.outputs = (new_output, ) + self.out.outputs

    def clear_logs(self):
        """ Clear the current logs """
        self.out.clear_output()

_widget_logger = None
_widget_log_handler = None

def get_widget_logger():
    if widgets is None:
        raise ModuleNotFoundError('ipywidgets')
    global _widget_logger, _widget_log_handler
    if _widget_logger is None:
        _widget_logger = logging.getLogger('EMAT.widget')
        _widget_log_handler = OutputWidgetHandler()
        _widget_log_handler.setFormatter(logging.Formatter('%(asctime)s  - [%(levelname)s] %(message)s'))
        _widget_logger.addHandler(_widget_log_handler)
        _widget_logger.setLevel(logging.INFO)
    return _widget_logger

def get_widget_log():
    global _widget_logger, _widget_log_handler
    get_widget_logger()
    return _widget_log_handler.out
