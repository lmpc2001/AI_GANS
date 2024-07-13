from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Concatenate

class Generator():
    def __init__(self, latent_dim, condition_dim):
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

        self.model = Model([input_latent, input_condition], x)