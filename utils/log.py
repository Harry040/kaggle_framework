import logging

level_dict = { 1: logging.INFO,2:logging.DEBUG}

logging.basicConfig(level=level_dict[1])

logger = logging.getLogger('app')
handler = logging.FileHandler('log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s -%(filename)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)
