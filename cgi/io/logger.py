from logging import Logger, getLogger, StreamHandler, Formatter, DEBUG


logger_count = -1


def make_logger(name: str) -> Logger:
    global logger_count
    logger_count += 1

    logger = getLogger(name + f'_{logger_count}')
    handler = StreamHandler()
    formatter = Formatter(
        '%(levelname)s [%(asctime)s | %(name)s]: %(message)s')

    handler.setFormatter(formatter)
    handler.setLevel(DEBUG)

    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
