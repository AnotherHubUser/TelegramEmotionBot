import logging
import contextvars
import sys
from pathlib import Path
from functools import wraps
from telegram import Update


update_id_ctx = contextvars.ContextVar('update_id', default='-')

def with_log_context(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        token = None

        update = kwargs.get('update')
        if isinstance(update, Update) and hasattr(update, 'update_id'):
            token = update_id_ctx.set(update.update_id)
        else:            
            for arg in args:
                if isinstance(arg, Update) and hasattr(arg, 'update_id'):
                    token = update_id_ctx.set(arg.update_id)
                    break
        try:
            return await func(*args, **kwargs)
        finally:
            if token:
                update_id_ctx.reset(token)
    return wrapper

class UpdateIdFilter(logging.Filter):
    def filter(self, record):
        record.update_id = update_id_ctx.get()
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

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt='%(asctime)s | [update:%(update_id)-8s] | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
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
    return logging.getLogger(name)
