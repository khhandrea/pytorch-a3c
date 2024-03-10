import numpy as np
from torch import tensor

class OnPolicyExperienceReplay():
    def __init__(self):
        self._size = 0
        self._keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self._reset()

    def _reset(self):
        for k in self._keys:
            setattr(self, k, [])
        self._size = 0

    def add_experience(self, state, action, reward, next_state, done):
        most_recent = (state, action, reward, next_state, done)
        for idx, key in enumerate(self._keys):
            getattr(self, key).append(most_recent[idx])
        self._size += 1

    def sample(self):
        batch = {}
        for key in self._keys:
            if key == 'dones':
                setattr(self, key, np.array((getattr(self, key)), dtype=np.float32))

            batch[key] = tensor(np.array(getattr(self, key)))

        self._reset()
        return batch

    def get_size(self):
        return self._size