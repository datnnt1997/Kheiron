from torch.optim.lr_scheduler import LambdaLR, StepLR


def get_linear_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_step_scheduler_with_warmup(optimizer, step_size, gamma, last_epoch=-1):
    return StepLR(optimizer, step_size, gamma=gamma, last_epoch=last_epoch)


class SchedulerNames:
    Linear = {'name': 'linear', 'cls': get_linear_scheduler_with_warmup, 'update_strategy': 'step'}
    StepLR = {'name': 'steplr', 'cls': get_step_scheduler_with_warmup, 'update_strategy': 'epoch'}

