from torch import nn, Tensor

class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU()
        )

    def policy(self, x: Tensor) -> Tensor:
        x = self.head(x)
        policy = self.actor(x)
        return policy

    def value(self, x: Tensor) -> Tensor:
        x = self.head(x)
        value = self.critic(x)
        return value.squeeze()