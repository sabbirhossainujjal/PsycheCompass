"""
Logging Configuration Module

Provides centralized logging setup for all components.
Each component (main, agents, memory, llm) gets its own log file.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with file and console handlers.
    
    Args:
        name: Logger name (e.g., 'main', 'agent_question_generator')
        log_file: Path to log file (e.g., 'logs/main.log')
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Default format string
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - writes all logs to file
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(formatter)
    
    # Console handler - writes INFO and above to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_session_start(logger: logging.Logger, session_id: str):
    """Log the start of a new session"""
    logger.info("="*70)
    logger.info(f"SESSION START - ID: {session_id}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*70)


def log_session_end(logger: logging.Logger, session_id: str, status: str = "completed"):
    """Log the end of a session"""
    logger.info("="*70)
    logger.info(f"SESSION END - ID: {session_id}")
    logger.info(f"Status: {status}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*70)


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(logger_name: str, level: int):
    """
    Change log level for a specific logger.
    
    Args:
        logger_name: Name of the logger
        level: New log level (e.g., logging.DEBUG)
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Update all handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)  # Always log everything to file
        else:
            handler.setLevel(level)


def clear_logs(log_directory: str = 'logs'):
    """
    Clear all log files in the specified directory.
    Use with caution!
    
    Args:
        log_directory: Directory containing log files
    """
    if not os.path.exists(log_directory):
        return
    
    for filename in os.listdir(log_directory):
        if filename.endswith('.log'):
            filepath = os.path.join(log_directory, filename)
            try:
                with open(filepath, 'w') as f:
                    f.write(f"# Log cleared at {datetime.now().isoformat()}\n")
            except Exception as e:
                print(f"Failed to clear {filepath}: {e}")


# Logging utility functions for common patterns

def log_dict(logger: logging.Logger, data: dict, level: int = logging.INFO, prefix: str = ""):
    """
    Log a dictionary in a readable format.
    
    Args:
        logger: Logger instance
        data: Dictionary to log
        level: Log level
        prefix: Optional prefix for the log message
    """
    if prefix:
        logger.log(level, f"{prefix}:")
    
    for key, value in data.items():
        logger.log(level, f"  {key}: {value}")


def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
    """
    Log an exception with context.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context about where/why the exception occurred
    """
    if context:
        logger.error(f"Exception in {context}: {str(exception)}", exc_info=True)
    else:
        logger.error(f"Exception occurred: {str(exception)}", exc_info=True)


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration: float,
    additional_info: dict = None
):
    """
    Log performance metrics for an operation.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        duration: Duration in seconds
        additional_info: Optional additional metrics
    """
    msg = f"Performance - {operation}: {duration:.3f}s"
    
    if additional_info:
        metrics = ", ".join([f"{k}={v}" for k, v in additional_info.items()])
        msg += f" ({metrics})"
    
    logger.info(msg)
