import numpy as np
import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from experience_replay import OnPolicyExperienceReplay
from network import ActorCriticNetwork

GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 1e-2

POLICY_SCALE = 0.0
VALUE_SCALE = 1.
ENTROPY_SCALE = 0.0

class A3C:
    def __init__(self, global_network: nn.Module):
        self.replay = OnPolicyExperienceReplay()
        self._global_network = global_network
        self._local_network = ActorCriticNetwork()
        self._optimizer = optim.SGD(self._global_network.parameters(), lr=LEARNING_RATE)
        self._gamma = GAMMA
        self._lambda = LAMBDA

    def get_action(self, observation: np.ndarray):
        observation_tensor = tensor(observation).view(1, -1)
        policy, _ = self._local_network(observation_tensor)
        distribution = Categorical(policy)
        action = distribution.sample()

        return action.numpy()
    
    def _calc_gaes(self, rewards, dones, v_preds):
        T = len(rewards)
        assert T + 1 == len(v_preds)
        gaes = torch.zeros_like(rewards, dtype=torch.float32)
        future_gae = torch.tensor(0.0, dtype=torch.float32)

        not_dones = 1 - dones
        for t in reversed(range(T)):
            delta = rewards[t] + GAMMA * v_preds[t + 1] * not_dones[t] - v_preds[t]
            gaes[t] = future_gae = delta + GAMMA * LAMBDA * not_dones[t] * future_gae
        return gaes

    def _get_gaes_v_targets(self, batch, v_preds):
        last_next_state = batch['next_states'][-1].unsqueeze(0)
        with torch.no_grad():
            _, last_v_pred = self._local_network(last_next_state)
        v_preds_all = torch.cat((v_preds, last_v_pred), dim=0)

        gaes = self._calc_gaes(batch['rewards'], batch['dones'], v_preds_all)
        v_targets = gaes + v_preds

        # Standardize
        if len(gaes) > 1:
            gaes = torch.squeeze((gaes - gaes.mean()) / (gaes.std() + 1e-8))

        return gaes, v_targets

    def train(self) -> tuple:
        batch = self.replay.sample()
        self._optimizer.zero_grad()

        policies, v_preds = self._local_network(batch['states'])

        print(policies[:3])
        distributions = Categorical(policies)
        log_probs = distributions.log_prob(batch['actions'])
        gaes, v_targets = self._get_gaes_v_targets(batch, v_preds.detach())

        policy_loss = -(gaes * log_probs).mean()
        value_loss = F.mse_loss(v_preds, v_targets)
        entropy = distributions.entropy().mean()
        loss = POLICY_SCALE * policy_loss + VALUE_SCALE * value_loss - ENTROPY_SCALE * entropy

        loss.backward()
        grad_norm = 0.
        for global_param, local_param in zip(self._global_network.parameters(), self._local_network.parameters()):
            global_param._grad = local_param.grad
            grad_norm += torch.norm(local_param.grad)**2
            local_param.grad= None
        grad_norm = np.sqrt(grad_norm)
        self._optimizer.step()

        result = (
            ('loss', loss.item()),
            ('gaes', gaes.mean().item()),
            ('log_probs', log_probs.mean().item()),
            ('policy_loss', policy_loss.item()),
            ('value_loss', value_loss.item()),
            ('entropy', entropy.item()),
            ('grad_norm', grad_norm.item())
        )

        return result

    def sync_network(self):
        self._local_network.load_state_dict(self._global_network.state_dict())