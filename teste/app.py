import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from Generator import Generator
from Discriminator import Discriminator
import sys
import numpy as np
from torch import nn
from configs import TRAIN_DATASET
from utils import load_dataset


# Parâmetros do Modelo
dim_entrada_gerador = 100
dim_saida_gerador = 9
dim_entrada_discriminador = 9

# Parâmetros de Treino
lr = 0.0002
beta1 = 0.5
num_epocas = 50
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o Dataset
df = load_dataset(TRAIN_DATASET)

# Normalizar os dados
df = (df - df.min()) / (df.max() - df.min())

# Criar o Dataset do PyTorch
class ContratosDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # data = self.df.iloc[:, 0].astype(str).str.strip().apply(pd.to_numeric, errors='coerce')
        # data = torch.tensor(data.values.astype(np.float32), dtype=torch.float32)
        data = self.df.iloc[idx, :-1].values.astype(np.float32)  # seleciona todas as colunas, exceto a última (que é a coluna de rótulos, se houver)
        data = torch.tensor(data, dtype=torch.float32)
        return data

dataset = ContratosDataset(df)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Criar o Generator e o Discriminator
gerador = Generator(dim_entrada_gerador, dim_saida_gerador).to(device)
discriminador = Discriminator(dim_entrada_discriminador).to(device)

# Criar as Funções de Perda e os Otimizadores
criterio_perda = nn.BCELoss()
otimizador_gerador = optim.Adam(gerador.parameters(), lr=lr, betas=(beta1, 0.999))
otimizador_discriminador = optim.Adam(discriminador.parameters(), lr=lr, betas=(beta1, 0.999))

# Treinar o Modelo
gerador.train()
discriminador.train()
for epoch in range(num_epocas):
    bar = tqdm(dataloader, file=sys.stdout)
    for real in bar:
        # Treinar o Discriminator
        discriminador.zero_grad()
        fake = gerador(torch.randn(real.shape[0], dim_entrada_gerador).to(device))
        pred_real = discriminador(real)
        pred_fake = discriminador(fake.detach())
        perda_discriminador = criterio_perda(pred_real, torch.ones_like(pred_real).to(device)) + criterio_perda(pred_fake, torch.zeros_like(pred_fake).to(device))
        perda_discriminador.backward()
        otimizador_discriminador.step()

        # # Treinar o Generator
        gerador.zero_grad()
        fake = gerador(torch.randn(real.shape[0], dim_entrada_gerador).to(device))
        pred_fake = discriminador(fake)
        perda_gerador = criterio_perda(pred_fake, torch.ones_like(pred_fake).to(device))
        perda_gerador.backward()
        otimizador_gerador.step()

        bar.set_description(f"Epoch [{epoch + 1}/{num_epocas}] | Perda D.: {perda_discriminador:.4f} | Perda G.: {perda_gerador:.4f}")

    print()

# Gerar Contratos Falsos
gerador.eval()
with torch.no_grad():
    fake = gerador(torch.randn(10, dim_entrada_gerador).to(device))
    fake = fake.cpu().numpy()

    print("Fake: ", fake)
    # Desnormalizar os dados
    fake = fake * (df.max().unsqueeze(1) - df.min().unsqueeze(1)) + torch.tensor(df.min().values).unsqueeze(1)



    # Arredondar os valores
    fake[:, 2] = fake[:, 2].astype(int)
    fake[:, 4] = fake[:, 4].astype(int)
    fake[:, 10] = fake[:, 10].astype(int)
    fake[:, 11] = fake[:, 11].astype(int)
    fake[:, 12] = fake[:, 12].astype(int)
    fake[:, 13] = fake[:, 13].astype(int)
    fake[:, 14] = fake[:, 14].astype(int)
    fake[:, 15] = fake[:, 15].astype(int)
    fake[:, 16] = fake[:, 16].astype(int)
    fake[:, 17] = fake[:, 17].astype(int)
    fake[:, 18] = fake[:, 18].astype(int)
    fake[:, 19] = fake[:, 19].astype(int)
    fake[:, 20] = fake[:, 20].astype(int)
    fake[:, 21] = fake[:, 21].astype(int)
    fake[:, 22] = fake[:, 22].astype(int)
    fake[:, 23] = fake[:, 23].astype(int)
    fake[:, 24] = fake[:, 24].astype(int)
    fake[:, 25] = fake[:, 25].astype(int)
    fake[:, 26] = fake[:, 26].astype(int)
    fake[:, 27] = fake[:, 27].astype(int)
    fake[:, 28] = fake[:, 28].astype(int)

    # Imprimir os contratos falsos
    for i in range(10):
        print(fake[i])
