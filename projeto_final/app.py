import torch
# import pandas as pd

from Generator import Generator
from Discriminator import Discriminator
from utils import load_dataset
from configs import TRAIN_DATASET, BATCH_SIZE, LR, NUM_EPOCHS
from PandasDataset import PandasDataset

generator = Generator()
discriminator = Discriminator()

data = load_dataset(TRAIN_DATASET)
training_set = PandasDataset(data)


torch.manual_seed(111)

train_loader = torch.utils.data.DataLoader(
    training_set, batch_size=BATCH_SIZE, shuffle=True
)

loss_function = torch.nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=LR)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=LR)


for epoch in range(NUM_EPOCHS):
    for n, (real_samples) in enumerate(train_loader):
        print(n, real_samples)
        # Data for training the discriminator
        real_samples_labels = torch.ones((BATCH_SIZE, 1))
        latent_space_samples = torch.randn((BATCH_SIZE, 2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((BATCH_SIZE, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels
        )
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((BATCH_SIZE, 2))

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if epoch % 10 == 0 and n == BATCH_SIZE - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
        # print("HI")
            
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)

generated_samples = generated_samples.detach()

print(generated_samples)
