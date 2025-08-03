import logging

from augment_service.config.settings import VERBOSITY

def setup_logger(name=__name__, level=logging.INFO, log_format="%(asctime)s - %(levelname)s - %(message)s"):
    verbosity_levels = {
        0: logging.ERROR,
        1: logging.INFO,
        2: logging.DEBUG
    }
    
    level = verbosity_levels.get(VERBOSITY, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )

    logging.getLogger('elasticsearch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('elastic_transport').setLevel(logging.WARNING)

    return logging.getLogger(name)

logger = setup_logger()