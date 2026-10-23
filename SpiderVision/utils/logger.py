"""
SpiderVision — Logger Utility
Structured, color-coded console logger.
"""
import logging
import sys
import time


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        ts    = time.strftime("%H:%M:%S")
        msg   = super().format(record)
        return f"{color}{self.BOLD}[{ts}] [{record.levelname:<8}]{self.RESET} {color}{record.getMessage()}{self.RESET}"


def get_logger(name: str = "SpiderVision") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ColorFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


log = get_logger()
