import numpy as np
import torch
from torch import nn, optim, tensor
from torch.distributions.categorical import Categorical

from experience_replay import OnPolicyExperienceReplay
from network import ActorCriticNetwork

STEP_MAX = 10000
GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 1e-4
VALUE_SCALE = 0.4
ENTROPY_SCALE = 0.0

class A3C:
    def __init__(self, global_network: nn.Module):
        self.replay = OnPolicyExperienceReplay()
        self._global_network = global_network
        self._local_network = ActorCriticNetwork()
        self._optimizer = optim.SGD(self._global_network.parameters(), lr=LEARNING_RATE)
        self._mse_loss = nn.MSELoss()
        self._gamma = GAMMA
        self._lambda = LAMBDA

    def get_action(self, observation: np.ndarray):
        observation_tensor = tensor(observation).view(1, -1)
        policy, _ = self._local_network(observation_tensor)
        distribution = Categorical(policy)
        action = distribution.sample()

        return action.numpy()

    def _get_gaes_v_targets(self, batch, v_preds):
        last_next_state = batch['next_states'][-1:]
        with torch.no_grad():
            _, last_v_pred = self._local_network(last_next_state)
        v_preds = v_preds.detach()

        v_preds_all = torch.cat((v_preds, last_v_pred), dim=0)

        T = len(batch['rewards'])
        gaes = torch.zeros_like(batch['rewards'], dtype=torch.float32)
        future_gae = tensor(0.0, dtype=torch.float32)
        not_dones = 1 - batch['dones']
        for t in reversed(range(T)):
            delta = batch['rewards'][t] + not_dones[t] * self._gamma * v_preds_all[t + 1] - v_preds_all[t]
            gaes[t] = future_gae = delta + self._gamma * self._lambda * not_dones[t] * future_gae

        v_targets = gaes.view(-1, 1) + v_preds

        # Standardize
        if len(gaes) > 1:
            gaes = torch.squeeze((gaes - gaes.mean()) / (gaes.std() + 1e-8))

        return gaes, v_targets

    def train(self):
        batch = self.replay.sample()

        policies, v_preds = self._local_network(batch['states'])
        distributions = Categorical(policies)
        log_probs = distributions.log_prob(batch['actions'])
        gaes, v_targets = self._get_gaes_v_targets(batch, v_preds)

        policy_loss = -(gaes * log_probs).mean()
        value_loss = self._mse_loss(v_targets, v_preds)
        entropy = distributions.entropy().mean()
        loss = policy_loss + VALUE_SCALE * value_loss - ENTROPY_SCALE * entropy

        self._optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(self._global_network.parameters(), self._local_network.parameters()):
            global_param._grad = local_param.grad
            self._optimizer.step()

        result = (
            ('loss', loss.item()),
            ('policy_loss', policy_loss.item()),
            ('value_loss', value_loss.item()),
            ('entropy', entropy.item())
        )

        return result

    def sync_network(self, global_network):
        self._local_network.load_state_dict(global_network.state_dict())