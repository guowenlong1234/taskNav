import logging


def emit_file_only(logger, level: int, message: str, exc_info=None) -> None:
    file_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    if len(file_handlers) == 0:
        logger.log(level, message, exc_info=exc_info)
        return

    record = logger.makeRecord(
        logger.name,
        level,
        __file__,
        0,
        message,
        args=(),
        exc_info=exc_info,
    )
    for handler in file_handlers:
        handler.handle(record)
