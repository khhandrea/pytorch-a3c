import numpy as np

class OnPolicyExperienceReplay():
    def __init__(self):
        self._size = 0
        self._replay_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self._reset()

    def _reset(self):
        for k in self._replay_keys:
            setattr(self, k, [])
        self._size = 0

    def add_experience(self, state, action, reward, next_state, done):
        most_recent = (state, action, reward, next_state, done)
        for idx, k in enumerate(self._replay_keys):
            getattr(self, k).append(most_recent[idx])
        self._size += 1

    def sample(self):
        batch = {k: np.array(getattr(self, k)) for k in self._replay_keys}
        self._reset()
        return batch

    def get_size(self):
        return self._size