import numpy as np
from cGan import cGan
from Discriminator import Discriminator
from Generator import Generator
from tensorflow.keras.optimizers import Adam
from utils import load_dataset2
from configs import TRAIN_DATASET
from sklearn.model_selection import train_test_split

[standardized_df, data] = load_dataset2(TRAIN_DATASET)

# Gerar rótulos condicionais (ajustar conforme necessário)
conditions = np.random.randint(0, 2, (data.shape[0], 10))  # 10 condições binárias

train_data, val_data, train_conditions, val_conditions = train_test_split(data, conditions, test_size=0.3, random_state=42)

latent_dim = 100 # Dimensão do vetor latente
condition_dim = train_conditions.shape[1]  # Número de condições
data_dim = train_data.shape[1] # Número de colunas com dados numéricos

# Montagem da cGan
generator = Generator(latent_dim, condition_dim, data_dim).model
discriminator = Discriminator(data_dim, condition_dim).model
discriminator.compile(loss='mean_squared_error', optimizer=Adam(0.0004, 0.5), metrics=['mse'])

cGan = cGan(generator, discriminator)

# Treino da cGan
cGan.train(generator, 
		   discriminator, 
		   epochs=1000, 
		   batch_size=64, 
		   latent_dim=latent_dim, 
		   condition_dim=condition_dim, 
		   train_data=train_data, 
		   train_conditions=train_conditions, 
		   val_data=val_data, 
		   val_conditions=val_conditions)