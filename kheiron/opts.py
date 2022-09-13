from typing import Optional, List
from dataclasses import dataclass, field, asdict

from enum import Enum

import datetime

import torch


@dataclass
class TrainingOptions:
    output_dir: str = field(default=f'trainer_output_{datetime.datetime.now().strftime("%H%M%S_%d%h%Y")}',
                            metadata={'help': 'The directory where model logs, predictions and checkpoints will be '
                                              'saved.'})
    task: str = field(default=f'text-classification',
                      metadata={'help': ('Task used for training model.',
                                         'Possible values are: ',
                                         '- `"text-classification"`: ... .Default'
                                         )})
    seed: int = field(default=42,
                      metadata={'help': 'Initialization of a random seed at the start of training.'})
    no_cuda: bool = field(default=False,
                          metadata={'help': 'Do not use CUDA even when it is available'})
    no_decay_param_names: List[str] = field(default=('bias', 'LayerNorm'),
                                            metadata={'help': ''})
    # Training Hyper-parameters
    epochs: int = field(default=5,
                        metadata={'help': ''})
    learning_rate: float = field(default=5e-5,
                                 metadata={'help': ''})
    train_batch_size: int = field(default=32,
                                  metadata={'help': ''})
    eval_batch_size: int = field(default=32,
                                 metadata={'help': ''})
    metric_for_best_model: str = field(default='loss',
                                       metadata={'help': 'The metric used to compare two models in order to determine '
                                                         'the best model.'})
    greater_is_better: bool = field(default=False,
                                    metadata={'help': ('Use in conjunction with `metric_for_best_model` to specify '
                                                       'whether or not better models should have a greater metric.')})
    # Optimizer
    optimizer_name: str = field(default='adamw',
                                metadata={'help': ''})
    weight_decay: float = field(default=0.0,
                                metadata={})
    adam_beta1: float = field(default=0.9,
                              metadata={'help': ''})
    adam_beta2: float = field(default=0.999,
                              metadata={'help': ''})
    adam_epsilon: float = field(default=1e-8,
                                metadata={'help': ''})
    max_grad_norm: float = field(default=1.0,
                                 metadata={'help': ''})
    momentum: float = field(default=0,
                            metadata={'help': 'momentum factor for SGD optimizer. (default: 0)'})

    # Learning rate Scheduler
    scheduler_name: str = field(default='linear',
                                metadata={'help': ''})
    warmup_proportion: float = field(default=0.0,
                                     metadata={'help': ''})
    warmup_steps: int = field(default=0,
                              metadata={'help': ''})
    decay_step: float = field(default=1.0,
                              metadata={'help': 'Period of learning rate decay for StepLR.'})
    gamma: float = field(default=0.1,
                         metadata={'help': 'Multiplicative factor of learning rate decay for StepLR.'})
    gradient_accumulation_steps: int = field(default=1,
                                             metadata={'help': ''})
    max_steps: int = field(default=0,
                           metadata={'help': ''})
    # Processor
    processor_num_workers: int = field(default=0,
                                       metadata={'help': ''})
    # Evaluation Hyper-parameters
    evaluation_strategy: str = field(default='epoch',
                                     metadata={'help': (
                                         'The evaluation strategy to adopt during training. ',
                                         'Possible values are: ',
                                         '- `"no"`: No evaluation is done during training.',
                                         '- `"steps"`: Evaluation is done (and logged) every `eval_steps`.',
                                         '- `"epoch"`: Evaluation is done at the end of each epoch.',)})
    eval_steps: Optional[int] = field(default=1,
                                      metadata={'help': 'Number of update steps between two evaluations if '
                                                        '`evaluation_strategy="steps"`.'})
    # Early Stopping
    early_stopping_steps: int = field(default=0,
                                      metadata={'help': ''})

    early_stopping_min_delta: float = field(default=0,
                                            metadata={'help': ''})

    # Other parameters
    track_metrics: bool = field(default=False,
                                metadata={'help': 'whether to enable track various metrics such as accuracy and log '
                                                  'loss on training or validation progress.'})

    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {k: v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    def __post_init__(self):
        if not self.no_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d
