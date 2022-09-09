"""Module defining various utilities."""
from kheiron.utils.logging import logger as LOGGER
from kheiron.utils.metrics import DefaultMetrics
from kheiron.utils.optimizers import OptimizerNames, get_lr_strings
from kheiron.utils.schedulers import SchedulerNames
from kheiron.utils.statistics import TrainingStats
from kheiron.utils.misc import set_ramdom_seed, get_spend_time

__all__ = ['LOGGER', 'DefaultMetrics', 'OptimizerNames', 'SchedulerNames', 'TrainingStats',
           'set_ramdom_seed',  'get_spend_time', 'get_lr_strings'
]