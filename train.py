import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from a3c import A3C

def train(global_network, total_step_max: int, step_max: int):
    env = gym.make('CartPole-v1')
    a3c = A3C(global_network)
    writer = SummaryWriter()

    total_step = 0
    start_step = 0

    episode = 0
    episode_return = 0
    state, _ = env.reset()

    while total_step < total_step_max:
        a3c.sync_network(global_network)

        action = a3c.get_action(state)[0]
        next_state, reward, terminated, truncated, _ = env.step(action)
        a3c.replay.add_experience(state, action, reward, next_state, terminated or truncated)
        episode_return += reward

        if (total_step - start_step == step_max) or (terminated or truncated):
            result = a3c.train()
            for tag, value in result:
                writer.add_scalar(f'loss/{tag}', value, total_step)

        if terminated or truncated:
            writer.add_scalar('return/return', episode_return, episode)
            episode_return = 0
            state, _ = env.reset()
            episode += 1

        total_step += 1
    env.close()