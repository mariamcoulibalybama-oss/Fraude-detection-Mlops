"""
Module de logging centralisé avec loguru.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message")
"""

import sys
from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    """Configure le logger global."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=level,
        colorize=True,
    )


def get_logger(name: str):
    """Retourne un logger contextualisé."""
    return logger.bind(name=name)


# Setup par défaut au chargement du module
setup_logger()
