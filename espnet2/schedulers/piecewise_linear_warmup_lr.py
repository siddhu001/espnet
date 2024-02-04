"""Piecewise linear warm up learning rate scheduler module."""
from typing import Union, List
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class PiecewiseLinearWarmupLR(_LRScheduler, AbsBatchStepScheduler):
    """The PiecewiseLinearWarmupLR scheduler

    This scheduler is similar to WarmupLR Scheduler except for following difference:

    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
        The warmup stage is linear.

    PiecewiseLinearWarmupLR:
        The warmup stage is piecewise linear.

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps_list: List[Union[int, float]] = [0, 25000],
        warmup_lr_list: List[float] = [0., 0.001],
        last_epoch: int = -1,
    ):
        assert check_argument_types()
        self.warmup_steps_list = warmup_steps_list
        self.warmup_lr_list = warmup_lr_list

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps_list={self.warmup_steps_list}, warmup_lr_list={self.warmup_lr_list})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            np.interp(
                step_num,
                self.warmup_steps_list,
                self.warmup_lr_list,
                right=lr * self.warmup_steps_list[-1]**0.5 * step_num**-0.5
            )
            for lr in self.base_lrs
        ]
