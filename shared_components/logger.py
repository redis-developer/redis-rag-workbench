import logging
import os

from dotenv import load_dotenv
from fastapi.logger import logger as fastapi_logger

load_dotenv()
log_level = os.environ.get("LOG_LEVEL", logging.DEBUG)
logger = logging.getLogger("uvicorn")
logger.setLevel(log_level)
fastapi_logger.setLevel(log_level)
