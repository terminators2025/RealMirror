# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
import logging
import sys
from colorama import Fore, Style, init

init()


class ColoredFormatter(logging.Formatter):
    """
    Custom log formatter that adds color to log levels.
    """

    format_str = "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

    FORMATS = {
        logging.DEBUG: str(Fore.CYAN) + format_str + str(Style.RESET_ALL),
        logging.INFO: str(Fore.GREEN) + format_str + str(Style.RESET_ALL),
        logging.WARNING: str(Fore.YELLOW) + format_str + str(Style.RESET_ALL),
        logging.ERROR: str(Fore.RED) + format_str + str(Style.RESET_ALL),
        logging.CRITICAL: str(Fore.RED) + str(Style.BRIGHT) + format_str + str(Style.RESET_ALL),
    }

    def format(self, record):
        try:
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)
        except Exception as e:
            # Fallback format if primary formatting fails
            return f"{Fore.RED}Log format error: {e} | Original message: {repr(record.msg)}{Style.RESET_ALL}"


class Logger:
    """
    A static logger class that provides colored, formatted logging without needing an instance.
    """

    # Initialize the logger at the class level. This code runs once when the module is imported.
    _logger = logging.getLogger("StaticColoredLogger")
    _logger.setLevel(logging.DEBUG)  # Set default level, e.g., logging.DEBUG to see debug messages

    # Ensure handlers are not added multiple times if the module is reloaded
    if not _logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        _logger.addHandler(console_handler)

    @staticmethod
    def _log(method, msg, *args, **kwargs):
        """Private helper to wrap logging calls with error handling."""
        # Add stacklevel=3 to look 3 frames up the stack to find the original caller.
        # Frame 1: _log() -> Frame 2: info() -> Frame 3: <actual caller>
        if "stacklevel" not in kwargs:
            kwargs["stacklevel"] = 3
        try:
            method(msg, *args, **kwargs)
        except Exception as e:
            try:
                Logger._logger.error(f"Logging error: {e} [Message: {repr(msg)}, Args: {repr(args)}]")
            except:
                # Last resort if the logger itself fails
                print("CRITICAL: Failure in logger's own error handling mechanism.")

    @staticmethod
    def debug(msg, *args, **kwargs):
        """Logs a message with level DEBUG on the root logger."""
        Logger._log(Logger._logger.debug, msg, *args, **kwargs)

    @staticmethod
    def info(msg, *args, **kwargs):
        """Logs a message with level INFO on the root logger."""
        Logger._log(Logger._logger.info, msg, *args, **kwargs)

    @staticmethod
    def warning(msg, *args, **kwargs):
        """Logs a message with level WARNING on the root logger."""
        Logger._log(Logger._logger.warning, msg, *args, **kwargs)

    @staticmethod
    def error(msg, *args, **kwargs):
        """Logs a message with level ERROR on the root logger."""
        Logger._log(Logger._logger.error, msg, *args, **kwargs)

    @staticmethod
    def critical(msg, *args, **kwargs):
        """Logs a message with level CRITICAL on the root logger."""
        Logger._log(Logger._logger.critical, msg, *args, **kwargs)

    @staticmethod
    def exception(msg, *args, **kwargs):
        """
        Logs a message with level ERROR on the root logger, with exception information.
        This method should be called from an exception handler.
        """
        Logger._log(Logger._logger.exception, msg, *args, **kwargs)
