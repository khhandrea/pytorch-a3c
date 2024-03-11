from torch import nn, Tensor

class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.head(x)
        policy = self.actor(x)
        value = self.critic(x)

        return policy, value
