"""
Logging utilities for the project.
"""

import logging
import os
from datetime import datetime
from typing import Optional


class ProjectLogger:
    """
    Custom logger for the diabetes prediction project.
    """
    
    def __init__(self, name: str = "EDRP", log_dir: str = "logs", level: int = logging.INFO):
        """
        Initialize the logger.
        
        Args:
            name (str): Logger name
            log_dir (str): Directory to save log files
            level (int): Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


# Global logger instance
logger = ProjectLogger()
