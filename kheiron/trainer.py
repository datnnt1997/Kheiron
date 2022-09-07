from typing import Optional, Any, List, Dict, Callable, TypeVar
from tqdm import tqdm

from kheiron import TrainingOptions
from kheiron.utils import DefaultMetrics, TrainingStats, OptimizerNames, SchedulerNames, LOGGER, set_ramdom_seed
from kheiron.constants import SavedFiles

from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader, RandomSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from datasets.arrow_dataset import Dataset
import os
import math
import torch
import time
import inspect

T = TypeVar('T')


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 args: TrainingOptions,
                 train_set: Optional[Dataset] = None,
                 eval_set: Optional[Dataset] = None,
                 optimizer: Optional[Optimizer] = None,
                 lr_scheduler: Optional[LambdaLR] = None,
                 eval_metrics: Optional[DefaultMetrics] = None,
                 collate_fn:  Callable[[List[T]], Any] = None):
        self.args = args
        self.optim = optimizer
        self.scheduler = lr_scheduler
        self.eval_metrics = eval_metrics
        self.model = model
        self.train_set = train_set
        self.eval_set = eval_set
        self.collate_fn = collate_fn

        # Create output directory if not exist
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        # Create training statistics
        self.stats = TrainingStats(metrics_tracking=args.track_metrics, log_dir=args.output_dir)

        # Set seed
        set_ramdom_seed(self.args.seed)

        # Move model to device
        self.model = model.to(self.args.device)
        self.signature_features = ["label", "label_ids"] + list(inspect.signature(self.model.forward).parameters.keys())
        self.ununsed_features = []

        # Set up the optimizer
        if self.optim is None:
            self.create_optimizer()

        if self.eval_metrics is None:
            self.eval_metrics = DefaultMetrics()

    def _remove_ununsed_features(self, feature: dict):
        for k in self.ununsed_features:
            del feature[k]

    def _wrap_collator_with_removed_unused_features(self, features: List[dict]):
        for feature in features:
            self._remove_ununsed_features(feature)
        if self.collate_fn is None:
            return features
        return self.collate_fn(features)

    def _save_training_step(self):
        # Save training arguments
        torch.save(self.args, os.path.join(self.args.output_dir, SavedFiles.TRAINING_ARGS_FILE))

        # Save training stats
        self.stats.save_to_json(os.path.join(self.args.output_dir, SavedFiles.TRAINING_STATS_FILE))

        # Save optimizer
        torch.save(self.optim.state_dict(), os.path.join(self.args.output_dir, SavedFiles.OPTIMIZER_FILE))

        # Save Scheduler
        torch.save(self.scheduler.state_dict(), os.path.join(self.args.output_dir, SavedFiles.SCHEDULER_FILE))

    def _save_model(self):
        state_dict = self.model.state_dict()
        LOGGER.info(f"Saving model checkpoint to {os.path.join(self.args.output_dir, SavedFiles.WEIGHT_MODEL_FILE)}")
        torch.save(state_dict, os.path.join(self.args.output_dir, SavedFiles.WEIGHT_MODEL_FILE))

    def _get_optimizer_parameter_names(self, model: nn.Module, parent_name: str = '') -> List[str]:
        if parent_name in self.args.no_decay_param_names:
            return []
        param_names = []
        for model_name, child_model in model.named_children():
            if model_name not in self.args.no_decay_param_names:
                param_names += [
                    f'{model_name}.{child_name}'
                    for child_name in self._get_optimizer_parameter_names(child_model, f'{parent_name}.{model_name}')
                ]
        param_names += [n for n in list(model._parameters.keys()) if n not in self.args.no_decay_param_names]
        return param_names

    def _preprocess_inputs(self, batch: dict):
        for k, v in batch.items():
            batch[k] = v.to(self.args.device)

    def _preprocess_logits(self, logits):
        if self.args.task == 'text-classification':
            return torch.argmax(logits, dim=-1)

    def _fit_one_epoch(self, process_bar):
        start_time = time.time()
        self.model.train()
        ep_loss = torch.tensor(0.0).to(self.args.device)

        for step_batch in process_bar:
            process_bar.desc = f'Epoch = {self.stats.get_value("curr_epoch")}/{self.stats.get_value("train_epochs")}; ' \
                               f'Global step = {self.stats.get_value("curr_global_step")}/{self.stats.get_value("max_steps")}'
            self.model.zero_grad()
            self._preprocess_inputs(step_batch)
            outputs = self.model(**step_batch)
            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    f"The 'loss' key was not found in the returned model results, only the following keys: "
                    f"{outputs.keys()}."
                )
            step_loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            if self.args.gradient_accumulation_steps > 1:
                step_loss = step_loss / self.args.gradient_accumulation_steps
            step_loss.backward()
            ep_loss += step_loss.detach()
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.args.max_grad_norm)
            self.optim.step()
            self.scheduler.step()
            self.stats.inc_value('curr_global_step')

    def _is_better_model(self, metrics: Dict, key_metric: str) -> bool:
        if self.args.greater_is_better \
                and self.stats.get_value('best_scores', float('-inf')) < metrics[key_metric]:
            return True
        elif not self.args.greater_is_better \
                and self.stats.get_value('best_scores', float('inf')) > metrics[key_metric]:
            return True
        else:
            return False

    def create_optimizer(self):
        decay_param_names = self._get_optimizer_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if n in decay_param_names],
                'lr': self.args.learning_rate, "weight_decay": self.args.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if n not in decay_param_names],
                'lr': self.args.learning_rate, "weight_decay": 0.0,
            }
        ]
        if self.args.optimizer_name == OptimizerNames.AdamW['name']:
            self.optim = OptimizerNames.AdamW['cls'](optimizer_grouped_parameters,
                                                     lr=self.args.learning_rate,
                                                     betas=(self.args.adam_beta1, self.args.adam_beta2),
                                                     eps=self.args.adam_epsilon)

    def create_scheduler(self, num_training_steps: int):
        warmup_steps = self.args.warmup_steps if self.args.warmup_steps > 0 \
            else math.ceil(num_training_steps * self.args.warmup_proportion)
        training_steps = self.args.epochs * num_training_steps
        if self.args.scheduler_name == SchedulerNames.Linear['name']:
            self.scheduler = SchedulerNames.Linear['cls'](optimizer=self.optim,
                                                          num_warmup_steps=warmup_steps,
                                                          num_training_steps=training_steps)

    def evaluate(self):
        if self.eval_set is None or \
                (not isinstance(self.eval_set, Dataset) and not isinstance(self.eval_set, TorchDataset)):
            raise TypeError(
                f"Expected type `torch.utils.data.Dataset` or `datasets.arrow_dataset.Dataset` for `eval_set`, "
                f"got '{type(self.eval_set)}' instead."
            )
        eval_iterator = DataLoader(self.eval_set,
                                   batch_size=self.args.eval_batch_size,
                                   collate_fn=self._wrap_collator_with_removed_unused_features,
                                   num_workers=self.args.processor_num_workers)
        num_examples = len(self.eval_set)
        len_iterator = len(eval_iterator)

        # Start fit model with trainning examples
        LOGGER.info("****** Evaluation Process ****** ")
        LOGGER.info(f"  Number of samples = {num_examples}")
        LOGGER.info(f"  Train batch size = {self.args.eval_batch_size}")

        eval_progress_bar = tqdm(eval_iterator,
                                 total=len_iterator,
                                 leave=True,
                                 position=0)
        self.model.eval()
        eval_loss = torch.tensor(0.0).to(self.args.device)
        golds, preds = [], []
        with torch.no_grad():
            for step_batch in eval_progress_bar:
                self._preprocess_inputs(step_batch)
                outputs = self.model(**step_batch)
                if isinstance(outputs, dict) and 'loss' not in outputs:
                    raise ValueError(
                        f"The 'loss' key was not found in the returned model results, only the following keys: "
                        f"{outputs.keys()}."
                    )
                step_loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                eval_loss += step_loss.detach()
                if 'labels' in step_batch:
                    golds.extend(step_batch['labels'].detach().cpu().tolist())
                if isinstance(outputs, dict) and 'logits' in outputs:
                    logits = self._preprocess_logits(outputs['logits'])
                    preds.extend(logits.detach().cpu().tolist())

        scores = self.eval_metrics(true_seqs=golds, pred_seqs=preds, task=self.args.task)
        metrics = {f"eval_{key}": score for key, score in scores.items()}
        metrics['eval_loss'] = eval_loss.item() / len_iterator

        # Update statistics with evaluation scores
        self.stats.update_eval_metrics(metrics)

        # Log result
        LOGGER.info(f"   Evaluation results: ")
        LOGGER.info(f'     Key metrics: Loss: {metrics["eval_loss"]}; {self.args.metric_for_best_model}: {metrics[f"eval_{self.args.metric_for_best_model}"]}')
        LOGGER.info(f'     Full metrics: {self.stats.get_eval_metrics_str()}')
        return metrics

    def train(self):
        if self.train_set is None or \
                (not isinstance(self.train_set, Dataset) and not isinstance(self.train_set, TorchDataset)):
            raise TypeError(
                f"Expected type `torch.utils.data.Dataset` or `datasets.arrow_dataset.Dataset` for `train_set`, "
                f"got '{type(self.train_set)}' instead."
            )
        self.ununsed_features = list(set(self.train_set[0].keys()) - set(self.signature_features))
        train_iterator = DataLoader(self.train_set,
                                    sampler=RandomSampler(self.train_set),
                                    batch_size=self.args.train_batch_size,
                                    collate_fn=self._wrap_collator_with_removed_unused_features,
                                    num_workers=self.args.processor_num_workers)
        num_examples = len(self.train_set)
        len_iterator = len(train_iterator)
        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(len_iterator // self.args.gradient_accumulation_steps, 1)
        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            num_train_epochs = math.ceil(self.args.max_steps / num_update_steps_per_epoch)
            num_train_samples = self.args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(self.args.epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.args.epochs)
            num_train_samples = num_examples * self.args.epochs

        # Set up learning rate scheduler
        if self.scheduler is None:
            self.create_scheduler(num_training_steps=max_steps)

        # Start fit model with trainning examples
        LOGGER.info("****** Trainning Process ****** ")
        LOGGER.info(f"  Training task            = {self.args.task}")
        LOGGER.info(f"  Number of samples        = {num_examples}")
        LOGGER.info(f"  Number of epochs         = {num_train_epochs}")
        LOGGER.info(f"  Train batch size         = {self.args.train_batch_size}")
        LOGGER.info(f"  Total optimization steps = {max_steps}")
        LOGGER.info(f"  Evaluation strategy      = {self.args.eval_steps}/{self.args.evaluation_strategy}")

        # Update Training statistics
        self.stats.set_value('max_steps', max_steps)
        self.stats.set_value('train_examples', num_examples)
        self.stats.set_value('train_epochs', num_train_epochs)
        self.stats.set_value('train_batch_size', self.args.train_batch_size)

        for curr_epoch in range(1, self.args.epochs):
            self.stats.set_value('curr_epoch', curr_epoch)
            trained_progress_bar = tqdm(train_iterator,
                                        total=num_update_steps_per_epoch,
                                        leave=True,
                                        position=0,
                                        desc=f'Epoch = {curr_epoch}/{num_train_epochs}; '
                                             f'Global step = {self.stats.get_value("curr_global_step")}/{max_steps}')
            self._fit_one_epoch(trained_progress_bar)
            if self.args.evaluation_strategy == 'epoch':
                metrics = self.evaluate()
                if self._is_better_model(metrics, f'eval_{self.args.metric_for_best_model}'):
                    self._save_model()
                    self._save_training_step()
