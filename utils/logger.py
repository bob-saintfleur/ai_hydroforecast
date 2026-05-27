import logging
import os
from datetime import datetime
from tqdm import tqdm
import getpass
user = getpass.getuser()

LOG_DIR = "LOGS"
os.makedirs(LOG_DIR, exist_ok=True)

# one log per launch
# TIMESTAMP = datetime.now().strftime("%Y%b%d_%H%M%S")
TIMESTAMP = datetime.now().strftime("%Y%b%d_%H")
LOG_FILE = os.path.join(LOG_DIR, f"{user.upper()}_RUN_{TIMESTAMP.upper()}.log")


# deal with TQDM
class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record=record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record=record)


class LazyFileHandler(logging.FileHandler):
    """ Delay file creation until first log emitted"""
    def __init__(self, filename, mode="a", encoding=None, delay=True):
        self._initialized = False
        self._filename = filename
        self._mode = mode
        self._encoding = encoding
        self._delay = delay
        self._stream = None
        super().__init__(filename=filename, mode=mode, encoding=encoding, delay=delay)

    def emit(self, record):
        if not self._initialized:
            self._open()
            self._initialized = True
        super().emit(record=record)


# set logger
def get_logger():
    logger_ = logging.getLogger("app_logger")
    logger_.setLevel(logging.DEBUG)

    if not logger_.hasHandlers():
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y.%m.%d %H:%M:%S")

        # console handler
        console_handler = TqdmLoggingHandler()

        console_handler.setFormatter(formatter)
        logger_.addHandler(console_handler)

        # File handler
        file_handler = LazyFileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger_.addHandler(file_handler)
    return logger_


logger = get_logger()
