import numpy as np
from cGan import cGan
from Discriminator import Discriminator
from Generator import Generator
from tensorflow.keras.optimizers import Adam
from utils import load_dataset2
from configs import TRAIN_DATASET


# Dados de treino (exemplo)
# data = np.random.normal(0, 1, (1000, 1))  # 1000 amostras de dados numéricos
[standardized_df, data] = load_dataset2(TRAIN_DATASET)
conditions = np.random.randint(0, 2, (446, 10))  # 10 condições binárias


latent_dim = 100
condition_dim = 10
data_dim = 1

# Montagem da cGan
generator = Generator(latent_dim, condition_dim).model
discriminator = Discriminator(data_dim, condition_dim).model
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

cGan = cGan(generator, discriminator)


# Treino da cGan
cGan.train(generator, discriminator, epochs=446, batch_size=64, latent_dim=latent_dim, condition_dim=condition_dim, data=data, conditions=conditions)