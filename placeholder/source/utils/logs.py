import sys
import logging


def _get_console_handler():
    """Get console handler

    Returns:
        logging.StreamHandler: Logs into stdout
    """
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(("%(asctime)s — %(name)s "
                                   "— %(levelname)s — %(message)s"))
    console_handler.setFormatter(formatter)

    return console_handler


def get_logger(name=__name__, log_level=logging.DEBUG):
    """Get logger

    Args:
        name (str): Logger name. Defaults to __name__.
        log_level (_type_): Logging level. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate outputs in Jypyter Notebook
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(_get_console_handler())
    logger.propagate = False

    return logger
