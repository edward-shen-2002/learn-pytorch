from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class Generator(nn.Module):
    #The generator takes an input of size latent_size, and will produce an output of size 784.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_size, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.layers(z)
        return x

class Discriminator(nn.Module):
    #The discriminator takes an input of size 784, and will produce an output of size 1.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its output
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.layers(x)
        return z

def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    #Trains both the generator and discriminator for one epoch on the training dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    discriminator.train()
    generator.train()
    num_batches = len(train_loader)
    loss_fn = nn.BCELoss()
    avg_generator_loss = 0
    avg_discriminator_loss = 0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)

        # 1. Discriminator
        # train discriminator on real data
        discriminator.zero_grad()
        real = discriminator(x)
        real_labels = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
        loss_d_real = loss_fn(real, real_labels)

        loss_d_real.backward()
        discriminator_optimizer.step()

        # train discriminator on fake data
        z = torch.randn(batch_size, latent_size).to(device)
        fake_in = generator(z)
        fake = discriminator(fake_in.detach())
        fake_labels = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)
        loss_d_fake = loss_fn(fake, fake_labels)

        loss_d_fake.backward()
        discriminator_optimizer.step()

        # 2. Generator
        generator.zero_grad()
        fake = discriminator(fake_in)
        loss_g = loss_fn(fake, real_labels)
        loss_g.backward()
        generator_optimizer.step()
        
        avg_discriminator_loss += (loss_d_real + loss_d_fake).item()
        avg_generator_loss += loss_g.item()
    avg_generator_loss /= num_batches
    avg_discriminator_loss /= num_batches
    return avg_generator_loss, avg_discriminator_loss

def test(generator, discriminator):
    #Runs both the generator and discriminator over the test dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    discriminator.eval()
    generator.eval()
    loss_fn = nn.BCELoss()
    num_batches = len(test_loader)
    avg_generator_loss = 0
    avg_discriminator_loss = 0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            real_labels = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
            fake_labels = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

            # test generator
            z = torch.randn(batch_size, latent_size, device=device)
            fake_x = generator(z)
            d_out = discriminator(fake_x)
            loss_g = loss_fn(d_out, real_labels)

            # test discriminator
            real = discriminator(x)
            loss_d_real = loss_fn(real, real_labels)
            fake = discriminator(fake_x)
            loss_d_fake = loss_fn(fake, fake_labels)

            avg_generator_loss += loss_g.item()
            avg_discriminator_loss += (loss_d_fake + loss_d_real).item()
    avg_generator_loss /= num_batches
    avg_discriminator_loss /= num_batches
    return avg_generator_loss, avg_discriminator_loss


epochs = 50

discriminator_avg_train_losses = []
discriminator_avg_test_losses = []
generator_avg_train_losses = []
generator_avg_test_losses = []

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator, discriminator_optimizer)
    generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

    discriminator_avg_train_losses.append(discriminator_avg_train_loss)
    generator_avg_train_losses.append(generator_avg_train_loss)
    discriminator_avg_test_losses.append(discriminator_avg_test_loss)
    generator_avg_test_losses.append(generator_avg_test_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = generator(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(discriminator_avg_train_losses)
plt.plot(generator_avg_train_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()

plt.plot(discriminator_avg_test_losses)
plt.plot(generator_avg_test_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()
