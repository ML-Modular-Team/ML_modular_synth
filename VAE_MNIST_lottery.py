# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:02:27 2020
@author: louis
"""



import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from pruning import Pruning_tool

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('using', device)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 40)
        self.fc22 = nn.Linear(400, 40)
        self.fc3 = nn.Linear(40, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def kl_anneal_function(anneal_function, step, k, x0):
    """ Beta update function
        
        Parameters
        ----------
        anneal_function : string
            What type of update (logisitc or linear)
        step : int
            Which step of the training
        k : float
            Coefficient of the logistic function
        x0 : float
            Delay of the logistic function or slope of the linear function
        Returns
        -------
        beta : float
            Weight of the KL divergence in the loss function 
        """
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta):
    """ Compute the loss function between recon_x (output of the VAE) 
    and x (input of the VAE)
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar)
    return BCE + beta*KLD


def train():
    global step
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(mnist_trainset):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        beta = kl_anneal_function('linear', step, 1, 10*len(mnist_trainset))
        print(beta)
        loss = loss_function(recon_batch, data, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        step += 1

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(mnist_trainset.dataset),
                100. * batch_idx / len(mnist_trainset),
                loss.item() / len(data)))
            plt_loss.append(loss.item() / len(data))  # For ploting loss

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(mnist_trainset.dataset)))


def test():
    test_step = step
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(mnist_testset):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            beta = kl_anneal_function('linear', test_step, 1,
                                      10*len(mnist_testset))
            test_loss += loss_function(recon_batch, data, mu, \
                                       logvar, beta).item()
            test_step += 1

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], \
                                recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.to(device), \
                           '/results/reconstruction_' + str(epoch) + '.png', \
                           nrow=n)

    test_loss /= len(mnist_testset.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


epochs = 20
batch_size = 64
log_interval = 100

mnist_trainset = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
                   batch_size=batch_size, shuffle=True)

mnist_testset = test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
                  batch_size=batch_size, shuffle=True)

model = VAE().to(device)
print(model.parameters)


# #local pruning
# for name, module in model.named_modules():
#     # prune 20% of connections in all 2D-conv layers
#     if isinstance(module, torch.nn.Conv2d):
#         prune.l1_unstructured(module, name='weight', amount=0.2)
#     # prune 40% of connections in all linear layers
#     elif isinstance(module, torch.nn.Linear):
#         prune.l1_unstructured(module, name='weight', amount=0.4)













    
pr = Pruning_tool()


#local pruning
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        pr.pruning(module,0.3)

        

pr.stats_pruning(model)


optimizer = optim.Adam(model.parameters(), lr=7e-4)
step = 0


# if __name__ == "__main__":
#     plt_loss = []

#     for epoch in range(1, epochs + 1):
#         train()
#         #test()
        
#         with torch.no_grad():
#             sample = torch.randn(64, 40).to(device)
#             sample = model.decode(sample).to(device)
#             save_image(sample.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')
    
