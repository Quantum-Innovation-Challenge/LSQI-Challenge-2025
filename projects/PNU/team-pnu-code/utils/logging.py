#!/usr/bin/env python3
"""
Logging utilities for PK/PD Modeling
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os


def setup_logging(log_dir: str, verbose: bool = False, run_name: Optional[str] = None) -> str:
    """Setup logging configuration"""
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file path
    if run_name:
        log_file = os.path.join(log_dir, f"{run_name}.log")
    else:
        log_file = os.path.join(log_dir, "training.log")
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)