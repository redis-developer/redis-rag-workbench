import logging
import os

from dotenv import load_dotenv
from fastapi.logger import logger as fastapi_logger

load_dotenv()
log_level = os.environ.get("LOG_LEVEL", logging.DEBUG)

logger = logging.getLogger("app")
logger.setLevel(log_level)

fastapi_logger.setLevel(log_level)

fastapi_cli = logging.getLogger("fastapi_cli")
fastapi_cli.setLevel(log_level)
logger.handlers = fastapi_cli.handlers
