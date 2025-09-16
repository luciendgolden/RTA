import logging

def configure_logging(verbosity: int = 1, log_format="%(asctime)s - %(levelname)s - %(message)s"):
    verbosity_levels = {
        0: logging.ERROR,
        1: logging.INFO,
        2: logging.DEBUG
    }
    
    level = verbosity_levels.get(verbosity, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )

    logging.getLogger('elasticsearch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('elastic_transport').setLevel(logging.WARNING)