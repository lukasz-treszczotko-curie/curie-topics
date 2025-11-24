import logging

import coloredlogs

# Define the log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FIELD_STYLES = {
    "asctime": {"color": "green"},
    "hostname": {"color": "magenta"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
    "programname": {"color": "cyan"},
}

LEVEL_STYLES = {
    "debug": {"color": "cyan"},
    "info": {"color": "white"},
    "warning": {"color": "yellow", "bold": True},
    "error": {"color": "red", "bold": True},
    "critical": {"color": "red", "bold": True, "background": "white"},
}


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up coloredlogs for the root logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured root logger
    """
    coloredlogs.install(
        level=level,
        fmt=LOG_FORMAT,
        field_styles=FIELD_STYLES,
        level_styles=LEVEL_STYLES,
    )
    return logging.getLogger()


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a named logger with coloredlogs configuration.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    coloredlogs.install(
        level=level,
        logger=logger,
        fmt=LOG_FORMAT,
        field_styles=FIELD_STYLES,
        level_styles=LEVEL_STYLES,
    )
    return logger
