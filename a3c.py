import numpy as np
import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from experience_replay import OnPolicyExperienceReplay
from network import ActorCriticNetwork
from utils import calc_returns, calc_gaes

GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 1e-3

POLICY_SCALE = 1.
VALUE_SCALE = 1.
ENTROPY_SCALE = 0.0

class A3C:
    def __init__(self, global_network: nn.Module):
        self.replay = OnPolicyExperienceReplay()
        self._global_network = global_network
        self._local_network = ActorCriticNetwork()
        self._optimizer = optim.Adam(self._global_network.parameters(), lr=LEARNING_RATE)

    def get_action(self, observation: np.ndarray):
        observation_tensor = tensor(observation, dtype=torch.float32).unsqueeze(0)
        policy = self._local_network.policy(observation_tensor)
        action = Categorical(policy).sample().item()
        return action

    def _get_gaes_target(self, batch, v_preds):
        with torch.no_grad():
            last_v_pred = self._local_network.value(batch['next_states'][-1:]).unsqueeze(0)
        v_preds_all = torch.cat((v_preds, last_v_pred), dim=0)
        gaes = calc_gaes(batch['rewards'], batch['dones'], v_preds_all, GAMMA, LAMBDA)
        v_target = gaes + v_preds
        return gaes, v_target

    def _get_advs_target(self, batch, v_preds):
        with torch.no_grad():
            last_v_pred = self._local_network.value(batch['next_states'][-1:])
        returns = calc_returns(batch['rewards'], batch['dones'], v_preds, GAMMA)
        advantages = returns - v_preds
        return advantages, returns

    def sync_network(self):
        self._local_network.load_state_dict(self._global_network.state_dict())

    def train(self) -> tuple[float]:
        batch = self.replay.sample()
        policies = self._local_network.policy(batch['states'])
        v_preds = self._local_network.value(batch['states'])
        distributions = Categorical(policies)
        log_probs = distributions.log_prob(batch['actions'])
        # advs, v_targets = self._get_advs_target(batch, v_preds.detach())
        advs, v_targets = self._get_gaes_target(batch, v_preds.detach())

        policy_loss = -(advs * log_probs).mean()
        value_loss = F.mse_loss(v_preds, v_targets)
        entropy = distributions.entropy().mean()
        loss = POLICY_SCALE * policy_loss + VALUE_SCALE * value_loss - ENTROPY_SCALE * entropy

        self._optimizer.zero_grad()
        loss.backward()
        grad_norm = 0.
        for global_param, local_param in zip(self._global_network.parameters(), self._local_network.parameters()):
            global_param._grad = local_param.grad
            grad_norm += torch.norm(local_param.grad)**2
            local_param.grad = None
        grad_norm = np.sqrt(grad_norm)
        self._optimizer.step()
        self.sync_network()

        result = (
            ('loss', loss.item()),
            ('gaes', advs.mean().item()),
            ('log_probs', log_probs.mean().item()),
            ('policy_loss', policy_loss.item()),
            ('value_loss', value_loss.item()),
            ('entropy', entropy.item()),
            ('grad_norm', grad_norm.item())
        )

        return result
