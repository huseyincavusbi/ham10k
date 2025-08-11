"""
Logging utilities for the skin cancer detection system.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from src.utils.config import settings

def setup_logging(
    log_level: str = settings.LOG_LEVEL,
    log_file: Optional[Path] = None,
    structured: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        structured: Whether to use structured logging with structlog
    
    Returns:
        Configured logger instance
    """
    
    # Configure basic logging
    log_level_obj = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level_obj,
        format=settings.LOG_FORMAT,
        handlers=handlers
    )
    
    if structured:
        # Configure structlog for better structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        return structlog.get_logger()
    else:
        return logging.getLogger(__name__)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)

# Create default logger
logger = setup_logging()
