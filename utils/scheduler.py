import math
import torch
import torch.optim as optim

class CosineAnnealingWithWarmup(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup, and support for resuming from checkpoint.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_steps (int): Total number of training steps (or epochs).
        max_lr (float): Max learning rate.
        min_lr (float): Min learning rate.
        warmup_steps (int): Number of steps for linear warmup.
        last_epoch (int): The index of last epoch (used when resuming). Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 total_steps: int,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 last_epoch: int = -1):
        assert warmup_steps < total_steps, "warmup_steps must be less than total_steps"

        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps

        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [
                base_lr + (self.max_lr - base_lr) * step / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        elif step < self.total_steps:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                for _ in self.base_lrs
            ]
        else:
            # After training is over, keep lr at min_lr
            return [self.min_lr for _ in self.base_lrs]

    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1

        lr_list = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lr_list):
            param_group['lr'] = lr

def scheduler_is(scheduler, optimizer, loaded_epoch):
    if scheduler == 'step':
        scheduler = scheduler_is(args.scheduler)
    elif scheduler == 'cosine':
        warmup_epochs=100#warmup
        total_epochs=1000#共有120个epoch，则用于cosine rate的一共有110个epoch
        lr = 0.00001
        n_t=0.5
        start_epoch = 0

        scheduler = CosineAnnealingWithWarmup(
                                        optimizer=optimizer,
                                        total_steps=total_epochs,
                                        max_lr=1e-4,
                                        min_lr=1e-6,
                                        warmup_steps=warmup_epochs,
                                        last_epoch=loaded_epoch - 1  # 如果你从 epoch n 加载，则传入 n-1
                                    )
    return scheduler