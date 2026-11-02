from __future__ import annotations

import logging
import sys

import structlog

from config import Config


def _broadcast_processor(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    from logging_broadcast import get_log_broadcast

    broadcast = get_log_broadcast()
    event_copy = dict(event_dict)
    if broadcast is not None:
        broadcast.publish(event_copy)

    from config import Config
    if Config.TDA_ENABLED:
        from tda.stream import get_tda_stream
        stream = get_tda_stream()
        if stream is not None:
            stream.ingest(dict(event_dict))

    return event_dict


def _safe_add_logger_name(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    record = event_dict.get("_record")
    if record is not None and getattr(record, "name", None) is not None:
        event_dict["logger"] = record.name
    elif logger is not None and hasattr(logger, "name") and logger.name is not None:
        event_dict["logger"] = logger.name
    else:
        event_dict.setdefault("logger", "unknown")
    return event_dict


def init_logging() -> None:
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        _safe_add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _broadcast_processor,
    ]

    if Config.LOG_FORMAT == "console":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            _safe_add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _broadcast_processor,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

    for noisy in ("uvicorn.access", "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
