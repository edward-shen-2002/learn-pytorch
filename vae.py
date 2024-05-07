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

kwargs = {'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # activations
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # encoder layers
        self.linear1 = nn.Linear(784, 400)
        self.linear2 = nn.Linear(400, latent_size)
        self.linear3 = nn.Linear(400, latent_size)
        # decoder layers
        self.linear4 = nn.Linear(latent_size, 400)
        self.linear5 = nn.Linear(400, 784)

    def encode(self, x):
        #The encoder will take an input of size 784, and will produce two vectors of size latent_size (corresponding to the coordinatewise means and log_variances)
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and have two linear output layers (no activations)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        mu = self.linear2(x)
        log_var = self.linear3(x)
        return mu, log_var

    def reparameterize(self, means, log_variances):
        #The reparameterization module lies between the encoder and the decoder
        #It takes in the coordinatewise means and log-variances from the encoder (each of dimension latent_size), and returns a sample from a Gaussian with the corresponding parameters
        vars = torch.exp(log_variances)
        z = means + vars * torch.randn(means.size()).to(device)
        return z

    def decode(self, z):
        #The decoder will take an input of size latent_size, and will produce an output of size 784
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
        x = self.relu(self.linear4(z))
        x = self.sigmoid(self.linear5(x))
        return x

    def forward(self, x):
        #Apply the VAE encoder, reparameterization, and decoder to an input of size 784
        #Returns an output image of size 784, as well as the means and log_variances, each of size latent_size (they will be needed when computing the loss)
        means, log_vars = self.encode(x)
        z = self.reparameterize(means, log_vars)
        output = self.decode(z)
        return output, means, log_vars

def vae_loss_function(reconstructed_x, x, means, log_variances):
    #Compute the VAE loss
    #The loss is a sum of two terms: reconstruction error and KL divergence
    #Use cross entropy loss between x and reconstructed_x for the reconstruction error (as opposed to L2 loss as discussed in lecture -- this is sometimes done for data in [0,1] for easier optimization)
    #The KL divergence is -1/2 * sum(1 + log_variances - means^2 - exp(log_variances)) as described in lecture
    #Returns loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    flatten = nn.Flatten()
    loss_fn = nn.BCELoss() # cross entropy loss caused blurry samples
    reconstruction_loss = ((flatten(x) - reconstructed_x)**2).sum()
    kl_div = -1/2 * torch.sum(1 + log_variances - means ** 2 - torch.exp(log_variances))
    loss = reconstruction_loss + kl_div
    return loss, reconstruction_loss


def train(model, optimizer):
    #Trains the VAE for one epoch on the training dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    num_batches = len(train_loader)
    model.train()
    avg_train_loss = 0
    avg_train_reconstruction_loss = 0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()

        re_x, means, log_vars = model(x)
        loss, re_loss = vae_loss_function(re_x, x, means, log_vars)

        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item()
        avg_train_reconstruction_loss += re_loss.item()
    avg_train_loss /= num_batches
    avg_train_reconstruction_loss /= num_batches
    return avg_train_loss, avg_train_reconstruction_loss

def test(model):
    #Runs the VAE on the test dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    model.eval()
    num_batches = len(test_loader)
    avg_test_loss = 0
    avg_test_reconstruction_loss = 0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            re_x, means, log_vars = model(x)
            loss, re_loss = vae_loss_function(re_x, x, means, log_vars)
            avg_test_loss += loss.item()
            avg_test_reconstruction_loss += re_loss.item()
    avg_test_loss /= num_batches
    avg_test_reconstruction_loss /= num_batches
    return avg_test_loss, avg_test_reconstruction_loss

epochs = 50
avg_train_losses = []
avg_train_reconstruction_losses = []
avg_test_losses = []
avg_test_reconstruction_losses = []

vae_model = VAE().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    avg_train_loss, avg_train_reconstruction_loss = train(vae_model, vae_optimizer)
    avg_test_loss, avg_test_reconstruction_loss = test(vae_model)
    
    avg_train_losses.append(avg_train_loss)
    avg_train_reconstruction_losses.append(avg_train_reconstruction_loss)
    avg_test_losses.append(avg_test_loss)
    avg_test_reconstruction_losses.append(avg_test_reconstruction_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(avg_train_reconstruction_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()

plt.plot(avg_test_reconstruction_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()
