from torch import nn

class Generator(nn.Module):
    def __init__(self, dim_entrada, dim_saida):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_entrada, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, dim_saida),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.model(x)
        return output