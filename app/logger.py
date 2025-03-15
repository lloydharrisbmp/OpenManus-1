"""
Centralized logging configuration for the application.
"""
import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

# Configure logging directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Log file paths
APP_LOG = LOG_DIR / "app.log"
ERROR_LOG = LOG_DIR / "error.log"
SECURITY_LOG = LOG_DIR / "security.log"

class StructuredLogger:
    """Custom logger that outputs structured JSON logs."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.setup_handlers()

    def setup_handlers(self):
        """Set up logging handlers with proper formatting."""
        # Clear any existing handlers
        self.logger.handlers = []

        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": %(message)s}'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handlers
        app_handler = logging.handlers.RotatingFileHandler(
            APP_LOG,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        app_handler.setFormatter(formatter)
        self.logger.addHandler(app_handler)

        # Error handler (ERROR and above)
        error_handler = logging.handlers.RotatingFileHandler(
            ERROR_LOG,
            maxBytes=10485760,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)

        # Security handler
        security_handler = logging.handlers.RotatingFileHandler(
            SECURITY_LOG,
            maxBytes=10485760,
            backupCount=5
        )
        security_handler.setFormatter(formatter)
        self.logger.addHandler(security_handler)

    def _format_message(self, message: Any, extra: Optional[Dict] = None) -> str:
        """Format message and extra data as JSON string."""
        log_data = {
            "message": str(message),
            "timestamp": datetime.utcnow().isoformat(),
        }
        if extra:
            log_data.update(extra)
        return json.dumps(log_data)

    def info(self, message: Any, extra: Optional[Dict] = None):
        """Log info level message."""
        self.logger.info(self._format_message(message, extra))

    def error(self, message: Any, extra: Optional[Dict] = None, exc_info=True):
        """Log error level message."""
        self.logger.error(self._format_message(message, extra), exc_info=exc_info)

    def warning(self, message: Any, extra: Optional[Dict] = None):
        """Log warning level message."""
        self.logger.warning(self._format_message(message, extra))

    def debug(self, message: Any, extra: Optional[Dict] = None):
        """Log debug level message."""
        self.logger.debug(self._format_message(message, extra))

    def critical(self, message: Any, extra: Optional[Dict] = None, exc_info=True):
        """Log critical level message."""
        self.logger.critical(self._format_message(message, extra), exc_info=exc_info)

    def security(self, message: Any, extra: Optional[Dict] = None):
        """Log security-related message."""
        security_data = {
            "type": "security",
            **(extra or {})
        }
        self.logger.info(self._format_message(message, security_data))

    def audit(self, action: str, user: str, resource: str, success: bool, extra: Optional[Dict] = None):
        """Log audit event."""
        audit_data = {
            "type": "audit",
            "action": action,
            "user": user,
            "resource": resource,
            "success": success,
            **(extra or {})
        }
        self.logger.info(self._format_message("Audit event", audit_data))

    def exception(self, message: Any, extra: Optional[Dict] = None):
        """Log exception with full stack trace."""
        self.logger.exception(self._format_message(message, extra))

# Create global logger instance
logger = StructuredLogger("financial_planner")

# Set default logging level from environment or default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.logger.setLevel(getattr(logging, LOG_LEVEL))

def get_logger(name: str) -> StructuredLogger:
    """Get a logger instance for a specific module."""
    return StructuredLogger(f"financial_planner.{name}")

# Example usage:
# from app.logger import logger, get_logger
# 
# # Using the global logger
# logger.info("Application started")
# logger.error("An error occurred", {"error_code": 500})
# 
# # Using a module-specific logger
# module_logger = get_logger("portfolio")
# module_logger.info("Portfolio analysis started", {"portfolio_id": "123"})
