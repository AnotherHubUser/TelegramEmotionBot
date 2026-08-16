import src.bot as bot
from src.config import TOKEN

from src.logger import setup_logger, get_logger

setup_logger(level='info', log_file='logs/log')
logger = get_logger(__name__)


def main():
    logger.info("creating bot")
    my_bot = bot.Bot(token=TOKEN)
    logger.info("bot created")
    
    try:
        my_bot.run()
    except KeyboardInterrupt:
        logger.info("bot stopped with control + C")
    except Exception as e:
        logger.critical(f"unexpected exception {e}")
    finally:
        logger.info("end of life")
    
if __name__ == "__main__":
    main()
