from torch import nn

class Discriminator(nn.Module):
    def __init__(self, 
                 in_features=2, out_features=256, probability=0.3,
                 in_features_2=256, out_features_2=128,
                 in_features_3=128, out_features_3=64,
                 in_features_4=64, out_features_4=1,):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(probability),
            nn.Linear(in_features_2, out_features_2),
            nn.ReLU(),
            nn.Dropout(probability),
            nn.Linear(in_features_3, out_features_3),
            nn.ReLU(),
            nn.Dropout(probability),
            nn.Linear(in_features_4, out_features_4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

# from tensorflow.keras.layers import Input, Dense, Concatenate
# from tensorflow.keras.models import Model

# from configs import NUM_CLASSES, NUM_FEATURES

# class Discriminator():
#     def __init__(self):
#         data_input = Input(shape=(NUM_FEATURES,))
#         class_input = Input(shape=(NUM_CLASSES,))
#         merged_input = Concatenate()([data_input, class_input])
#         hidden = Dense(128, activation='relu')(merged_input)
#         output = Dense(1, activation='sigmoid')(hidden)
#         self.model = Model(inputs=[data_input, class_input], outputs=output)