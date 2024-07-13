from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class cGan():
    def __init__(self, generator: Model, discriminator: Model):
        discriminator.trainable = False
        latent_dim = generator.input[0].shape[1]
        condition_dim = generator.input[1].shape[1]
        
        input_latent = Input(shape=(latent_dim,))
        input_condition = Input(shape=(condition_dim,))

        generated_data = generator([input_latent, input_condition])
        validity = discriminator([generated_data, input_condition])
        
        self.model = Model([input_latent, input_condition], validity)
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    def train(self, generator, discriminator, epochs, batch_size, latent_dim, condition_dim, train_data, train_conditions, val_data, val_conditions):
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            idx = np.random.randint(0, train_data.shape[0], half_batch)
            real_data = train_data[idx]
            real_conditions = train_conditions[idx]
            
            noise = np.random.normal(0, 1, (half_batch, latent_dim))
            generated_data = generator.predict([noise, real_conditions])
            
            d_loss_real = discriminator.train_on_batch([real_data, real_conditions], np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch([generated_data, real_conditions], np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            sampled_conditions = np.random.randint(0, 2, (batch_size, condition_dim))  # Assumes binary conditions
            valid_y = np.array([1] * batch_size)
            
            g_loss = self.model.train_on_batch([noise, sampled_conditions], valid_y)

            if epoch % 100 == 0:
                print(f"\nEpoch {epoch}/{epochs} | D Loss: {d_loss[0]} | D Accuracy: {d_loss[1]} | G Loss: {g_loss}")
           
            # Avaliação do modelo em dados de validação
            if epoch % 100 == 0:
                val_noise = np.random.normal(0, 1, (val_data.shape[0], latent_dim))
                val_generated_data = generator.predict([val_noise, val_conditions])
                val_d_loss_real = discriminator.evaluate([val_data, val_conditions], np.ones((val_data.shape[0], 1)), verbose=0)
                val_d_loss_fake = discriminator.evaluate([val_generated_data, val_conditions], np.zeros((val_data.shape[0], 1)), verbose=0)
                val_d_loss = 0.5 * np.add(val_d_loss_real, val_d_loss_fake)
                print(f"Validation | D Loss: {val_d_loss[0]} | D Accuracy: {val_d_loss[1]}\n\n")
            
            # print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss} | G Loss: {g_loss} | Val Loss: {val_loss}")
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss} | G Loss: {g_loss}\n")