from torch import nn

class Generator(nn.Module):
    def __init__(self, 
                 in_features=2, out_features=16,
                 in_features_2=16, out_features_2=32,
                 in_features_3=32, out_features_3=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(in_features_2, out_features_2),
            nn.ReLU(),
            nn.Linear(in_features_3, out_features_3),
        )

    def forward(self, x):
        output = self.model(x)
        return output

# from tensorflow.keras.layers import Input, Dense, Concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# from configs import NOISE_DIM, NUM_CLASSES, NUM_FEATURES

# class Generator():
#     def __init__(self):
#         noise_input = Input(shape=(NOISE_DIM,))
#         class_input = Input(shape=(NUM_CLASSES,))
#         merged_input = Concatenate()([noise_input, class_input])
#         hidden = Dense(128, activation='relu')(merged_input)
#         output = Dense(NUM_FEATURES, activation='linear')(hidden)
#         self.model = Model(inputs=[noise_input, class_input], outputs=output)