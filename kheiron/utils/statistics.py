from typing import Union
from datetime import datetime
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

import os
import json
import time

class TrainingStats:
    def __init__(self,
                 metrics_tracking=False,
                 log_dir: Union[str or os.PathLike] = f'trainer_output_{datetime.now().strftime("%H%M%S_%d%h%Y")}'):
        default_stats = {
            # Training progress
            'curr_epoch': 0,
            'curr_global_step': 0,
            'max_steps': 0,
            'train_examples': 0,
            'train_epochs': 0,
            'train_batch_size': 0,
            'train_metrics': defaultdict(list) if metrics_tracking else {},
            'closed_reason': None,
            # Evaluate progress
            'eval_examples': 0,
            'eval_batch_size': 0,
            'total_eval_steps': 0,
            'evaluation_strategy': None,
            'metric_for_best_model': 'loss',
            'eval_metrics': defaultdict(list) if metrics_tracking else {}

        }
        self._stats = default_stats
        self.metrics_tracking = metrics_tracking
        if metrics_tracking:
            self.tracking_board = SummaryWriter(log_dir=os.path.join(log_dir, 'logs'))
        self.cached_time = time.time()

    def set_stats(self, stats):
        self._stats = stats

    def get_value(self, key, default=None):
        return self._stats.get(key, default)

    def set_value(self, key, value):
        self._stats[key] = value

    def inc_value(self, key, count=1, start=0):
        d = self._stats
        if d.setdefault(key, start) is None:
            d[key] = count
        else:
            d[key] = d.setdefault(key, start) + count

    def update_time(self):
        self.cached_time = time.time()

    def clear_stats(self):
        self._stats.clear()

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(self._stats, indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    def load_from_json(self, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        self._stats = json.loads(text)

    def get_evaluation_step(self):
        if self._stats['evaluation_strategy'] == 'epoch':
            return self.get_value('curr_epoch')
        else:
            return self.get_value('curr_global_step')

    def _update_metrics(self, metrics: dict, metrics_name: str):
        if self.metrics_tracking:
            for k, v in metrics.items():
                self.tracking_board.add_scalar(tag=k,
                                               scalar_value=v,
                                               global_step=self.get_evaluation_step())
                self._stats[metrics_name][k].append(v)
        else:
            self.set_value(metrics_name, metrics)

    def _metrics_to_str(self, metrics_name):
        metrics_str = ''
        for k, v in self._stats[metrics_name].items():
            curr_value = v[-1] if self.metrics_tracking else v
            metrics_str += f'{k} = {curr_value:.4f}; '
        return metrics_str

    def update_train_metrics(self, metrics: dict):
        self._update_metrics(metrics, metrics_name='train_metrics')

    def update_eval_metrics(self, metrics: dict):
        self._update_metrics(metrics, metrics_name='eval_metrics')

    def get_train_metrics_str(self) -> str:
        return self._metrics_to_str(metrics_name='train_metrics')

    def get_eval_metrics_str(self):
        return self._metrics_to_str(metrics_name='eval_metrics')
