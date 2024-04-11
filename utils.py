import torch

def calc_gaes(rewards, dones, v_preds, gamma, lmbda):
    T = len(rewards)
    assert T + 1 == len(v_preds)
    gaes = torch.zeros_like(rewards, dtype=torch.float32)
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    future_gae = torch.tensor(0.0, dtype=torch.float32)
    G = v_preds[-1]

    not_dones = 1 - dones
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * v_preds[t + 1] * not_dones[t] - v_preds[t]
        gaes[t] = future_gae = delta + gamma * lmbda * not_dones[t] * future_gae
        returns[t] = G = rewards[t] + gamma * not_dones[t] * G
    return gaes, returns

def calc_nstep_returns(rewards, dones, next_v_pred, n, gamma):
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    future_ret = next_v_pred
    not_dones = 1 - dones
    for t in reversed(range(n)):
        returns[t] = future_ret = rewards[t] + gamma * future_ret * not_dones[t]

    return returns

def calc_td_returns(rewards, dones, next_v_preds, gamma):
    not_dones = 1 - dones
    returns = rewards + gamma * next_v_preds * not_dones
    return returns.to(torch.float32)