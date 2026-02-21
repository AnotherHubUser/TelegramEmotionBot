import logging
import sys
import threading
from pathlib import Path
from functools import wraps
from telegram import Update


_context = threading.local()

def set_update_id(update_id):
    _context.update_id = update_id

def get_update_id():
    return getattr(_context, 'update_id', '-')

def clear_update_id():
    if hasattr(_context, 'update_id'):
        delattr(_context, 'update_id')

def with_log_context(func):
    @wraps(func)
    async def wrapper(self, update, *args, **kwargs):
        if update and hasattr(update, 'update_id'):
            set_update_id(update.update_id)
        try:
            return await func(self, update, *args, **kwargs)
        finally:
            clear_update_id()
    return wrapper

class UpdateIdFilter(logging.Filter):
    def filter(self, record):
        record.update_id = get_update_id()
        return True

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

def setup_logger(name: str = None, level: str = 'info', log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
    
    formatter = logging.Formatter(
        fmt='%(asctime)s | [update:%(update_id)-8s] | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(UpdateIdFilter())
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  
        file_handler.addFilter(UpdateIdFilter())
        logger.addHandler(file_handler)

    for lib in ["httpcore", "httpx", "telegram", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    return setup_logger(name)
