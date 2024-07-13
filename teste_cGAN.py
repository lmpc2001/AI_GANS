import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Criação do Gerador
def build_generator(latent_dim, condition_dim):
    input_latent = Input(shape=(latent_dim,))
    input_condition = Input(shape=(condition_dim,))
    merged_input = Concatenate()([input_latent, input_condition])

    x = Dense(128)(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1, activation='linear')(x)  # Assume that the output is a single numeric value

    model = Model([input_latent, input_condition], x)
    return model

# Construção do Discriminador
def build_discriminator(data_dim, condition_dim):
    input_data = Input(shape=(data_dim,))
    input_condition = Input(shape=(condition_dim,))
    merged_input = Concatenate()([input_data, input_condition])

    x = Dense(512)(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model([input_data, input_condition], x)
    return model

# Construção do Discriminador
def build_cgan(generator, discriminator):
    discriminator.trainable = False
    latent_dim = generator.input[0].shape[1]
    condition_dim = generator.input[1].shape[1]
    
    input_latent = Input(shape=(latent_dim,))
    input_condition = Input(shape=(condition_dim,))
    
    generated_data = generator([input_latent, input_condition])
    validity = discriminator([generated_data, input_condition])
    
    model = Model([input_latent, input_condition], validity)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    
    return model

# Função de treino da cGAN
def train(generator, discriminator, combined, epochs, batch_size, latent_dim, condition_dim, data, conditions):
    half_batch = int(batch_size / 2)
    
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_data = data[idx]
        real_conditions = conditions[idx]
        
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        generated_data = generator.predict([noise, real_conditions])
        
        d_loss_real = discriminator.train_on_batch([real_data, real_conditions], np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch([generated_data, real_conditions], np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_conditions = np.random.randint(0, 2, (batch_size, condition_dim))  # Assumes binary conditions
        valid_y = np.array([1] * batch_size)
        
        g_loss = combined.train_on_batch([noise, sampled_conditions], valid_y)
        
        print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss} | G Loss: {g_loss}")

# Dados de treino (exemplo)
data = np.random.normal(0, 1, (1000, 1))  # 1000 amostras de dados numéricos
conditions = np.random.randint(0, 2, (1000, 10))  # 10 condições binárias

latent_dim = 100
condition_dim = 10
data_dim = 1

generator = build_generator(latent_dim, condition_dim)
discriminator = build_discriminator(data_dim, condition_dim)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

combined = build_cgan(generator, discriminator)

train(generator, discriminator, combined, epochs=10000, batch_size=64, latent_dim=latent_dim, condition_dim=condition_dim, data=data, conditions=conditions)
