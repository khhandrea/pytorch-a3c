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

    def get_action(self, observation: np.ndarray):
        observation_tensor = tensor(observation).view(1, -1)
        policy, _ = self._local_network(observation_tensor)
        distribution = Categorical(policy)
        action = distribution.sample()

        return action.numpy()
    
    def _get_gae_advantages(self, rewards, dones, v_preds, gamma, lam):
        T = len(rewards)
        gaes = torch.zeros_like(rewards)
        future_gae = tensor(0.0, dtype=float)
        not_dones = 1 - dones
        for t in reversed(range(T)):
            delta = rewards[t] + not_dones[t] * gamma * v_preds[t + 1] - v_preds[t]
            gaes[t] = future_gae = delta + gamma * lam * not_dones[t] * future_gae
        return gaes.view(-1, 1)

    def sync_network(self, global_network):
        self._local_network.load_state_dict(global_network.state_dict())

    def train(self):
        batch = self.replay.sample()

        states_tensor = tensor(batch['states'])
        actions_tensor = tensor(batch['actions'])
        rewards_tensor = tensor(batch['rewards'])
        last_next_state_tensor = tensor(batch['next_states'][-1:])
        dones_tensor = tensor([1 if done else 0 for done in batch['dones']])
        states_tensor = torch.cat((states_tensor, last_next_state_tensor), dim=0)

        policies, values = self._local_network(states_tensor)
        policies = policies[:-1]
        distributions = Categorical(policies)
        log_probs = distributions.log_prob(actions_tensor)

        values = values.detach()
        advantages = self._get_gae_advantages(rewards_tensor, dones_tensor, values, GAMMA, LAMBDA)

        standardized_adv = torch.squeeze((advantages - advantages.mean()) / (advantages.std() + 1e-08))
        policy_loss = -(standardized_adv * log_probs).mean()
        value_loss = torch.square(advantages).mean()
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