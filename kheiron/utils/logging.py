from loguru import logger

import sys


logger.remove()
logger.add(sys.stderr,
           format='<g>{time:ddd, DD MMM YYYY HH:mm:SS}</g> '
                  '| <m><b>Kheiron</b></m> '
                  '| <e><b><i>{level}</i></b></e> '
                  ': {message}',
           level='INFO')