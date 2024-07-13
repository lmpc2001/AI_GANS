from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

class Discriminator():
    def __init__(self, data_dim, condition_dim):
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

        self.model = Model([input_data, input_condition], x)
