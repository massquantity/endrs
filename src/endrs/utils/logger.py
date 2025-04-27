import sys
from pathlib import Path

from loguru import logger

logger.is_ready = False

# def custom_formatter(record: dict[str, Any]) -> str:
#    return record["extra"]["metrics"]


def setup_logger(
    log_level: str = "INFO",
    log_dir: str = "endrs_logs",
    log_file: str = "app.log",
    metrics_file: str = "metrics.log",
    clear_files: bool = False,
):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    if clear_files:
        (log_path / log_file).write_text("")
        (log_path / metrics_file).write_text("")

    logger.remove()
    add_normal_logger(log_level, log_path, log_file)
    add_metrics_logger(log_level, log_path, metrics_file)

    logger.bind(task="normal").info(f"Logging setup completed!")
    logger.is_ready = True


def add_normal_logger(log_level: str, log_path: Path, log_file: str):
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        filter=lambda record: record["extra"]["task"] == "normal",
    )

    logger.add(
        log_path / log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        filter=lambda record: record["extra"]["task"] == "normal",
        rotation="5 MB",
    )


def add_metrics_logger(log_level: str, log_path: Path, metrics_file: str):
    logger.add(
        sys.stdout,
        level=log_level,
        format="{message}",
        filter=lambda record: record["extra"]["task"] == "metrics",
    )

    logger.add(
        log_path / metrics_file,
        level=log_level,
        format="{message}",
        filter=lambda record: record["extra"]["task"] == "metrics_file",
    )


def remove_logger():
    logger.is_ready = False


def is_logger_ready() -> bool:
    return logger.is_ready
