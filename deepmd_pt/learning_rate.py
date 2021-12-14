import numpy as np


class LearningRateExp(object):

    def __init__(self, start_lr, stop_lr, decay_steps, stop_steps, **kwargs):
        '''Construct an exponential-decayed learning rate.

        Args:
        - start_lr: Initial learning rate.
        - stop_lr: Learning rate at the last step.
        - decay_steps: Decay learning rate every N steps.
        - stop_steps: When is the last step.
        '''
        self.start_lr = start_lr
        self.decay_steps = decay_steps
        self.decay_rate = np.exp(np.log(stop_lr / start_lr) / (stop_steps / decay_steps))

    def value(self, step):
        '''Get the learning rate at the given step.'''
        return self.start_lr * np.power(self.decay_rate, step // self.decay_steps)
