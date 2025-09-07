import logging
import sys
import io
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = "workflow.log"):
    """
    Configure logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure handlers (ensure UTF-8 everywhere)
    file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")

    # Wrap stdout with UTF-8 if needed to avoid UnicodeEncodeError on Windows consoles
    stdout_stream = sys.stdout
    try:
        encoding_name = getattr(sys.stdout, "encoding", None) or ""
        if encoding_name.lower() != "utf-8":
            stdout_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        # Fallback to default stdout if wrapping fails
        stdout_stream = sys.stdout

    stream_handler = logging.StreamHandler(stdout_stream)

    handlers = [file_handler, stream_handler]
    
    # Setup basic configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # Set specific loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("github").setLevel(logging.WARNING)
    
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)