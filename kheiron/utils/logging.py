from loguru import logger

import sys


logger.remove()
logger.add(sys.stderr,
           format='<g>{time:ddd, DD MMM YYYY HH:mm:SS}</g> '
                  '| <m><b>Trainer</b></m> '
                  '| <e><b><i>{level}</i></b></e> '
                  ': <w>{message}</w>',
           level='INFO')