import numpy as np
from sklearn.model_selection import train_test_split

from configs import TRAIN_DATASET
from utils import load_dataset2

[standardized_df, data] = load_dataset2(TRAIN_DATASET)

# Gerar rótulos condicionais (ajustar conforme necessário)
conditions = np.random.randint(0, 2, (data.shape[0], 15))  # 15 condições binárias

# Divisão do dataset em dados de treino e de validação
train_data, val_data, train_conditions, val_conditions = train_test_split(data, conditions, test_size=0.2, train_size=0.7, random_state=42)


latent_dim = 100 # Dimensão do vetor latente
condition_dim = train_conditions.shape[1]  # Número de condições
data_dim = train_data.shape[1] # Número de colunas com dados numéricos


print("Train data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)
print("Train conditions shape:", train_conditions.shape)
print("Validation conditions shape:", val_conditions.shape)
print("Data dimension:", data_dim)
print("Condition dimension:", condition_dim)