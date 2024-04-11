import numpy as np
import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from experience_replay import OnPolicyExperienceReplay
from network import ActorCriticNetwork
from utils import calc_gaes, calc_nstep_returns, calc_td_returns

GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 1e-4

REWARD_SCALE = 0.01
POLICY_SCALE = 1.
VALUE_SCALE = 5.
ENTROPY_SCALE = 0.0

class A3C:
    def __init__(self, global_network: nn.Module):
        self.replay = OnPolicyExperienceReplay()
        self._global_network = global_network
        self._local_network = ActorCriticNetwork()
        self._optimizer = optim.Adam(self._global_network.parameters(), lr=LEARNING_RATE)

    def get_action(self, observation: np.ndarray):
        observation_tensor = tensor(observation).view(1, -1)
        policy, _ = self._local_network(observation_tensor)
        distribution = Categorical(policy)
        action = distribution.sample()

        return action.numpy()
    

    def _get_gaes_v_targets(self, batch, v_preds):
        last_next_state = batch['next_states'][-1].unsqueeze(0)
        with torch.no_grad():
            _, last_v_pred = self._local_network(last_next_state)
        v_preds_all = torch.cat((v_preds, last_v_pred), dim=0)

        gaes, returns = calc_gaes(batch['rewards'], batch['dones'], v_preds_all, GAMMA, LAMBDA)
        # v_targets = gaes + v_preds
        v_targets = returns

        # Standardize
        if len(gaes) > 1:
            gaes = torch.squeeze((gaes - gaes.mean()) / (gaes.std() + 1e-8))

        return gaes, v_targets

    
    def _get_nstep_advs_v_target(self, batch, v_preds):
        last_next_state = batch['next_states'][-1].unsqueeze(0)
        with torch.no_grad():
            _, last_v_pred = self._local_network(last_next_state)
        returns = calc_nstep_returns(batch['rewards'], batch['dones'], last_v_pred, len(batch['rewards']), GAMMA)
        advs = returns - v_preds
        v_targets = returns

        return advs, v_targets
    
    def _get_tds_v_target(self, batch, v_preds):
        _, v_preds = self._local_network(batch['states'])
        _, next_v_preds = self._local_network(batch['next_states'])
        returns = calc_td_returns(batch['rewards'], batch['dones'], next_v_preds.detach(), GAMMA)
        targets = (batch['rewards'] + GAMMA * next_v_preds.detach() * (1 - batch['dones'])).to(torch.float32)
        advantages = targets - v_preds.detach()
        return advantages, targets

    def train(self) -> tuple:
        batch = self.replay.sample()

        policies, v_preds = self._local_network(batch['states'])

        distributions = Categorical(policies)
        log_probs = distributions.log_prob(batch['actions'])
        # advs, v_targets = self._get_gaes_v_targets(batch, v_preds.detach())
        # advs, v_targets = self._get_nstep_advs_v_target(batch, v_preds.detach())
        advs, v_targets = self._get_tds_v_target(batch, v_preds.detach())

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

        result = (('loss', loss.item()),
            ('gaes', advs.mean().item()),
            ('log_probs', log_probs.mean().item()),
            ('policy_loss', policy_loss.item()),
            ('value_loss', value_loss.item()),
            ('entropy', entropy.item()),
            ('grad_norm', grad_norm.item())
        )

        return result

    def sync_network(self):
        self._local_network.load_state_dict(self._global_network.state_dict())