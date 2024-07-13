from torch import nn

class Discriminator(nn.Module):
    def __init__(self, dim_entrada):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_entrada, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output