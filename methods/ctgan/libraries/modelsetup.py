import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.net = nn.Sequential()

       # Durchlaufen der Layers, bis auf das letzte
        for i in range(len(self.config.layers) - 1):
            self.net.append(nn.Linear(self.config.layers[i], self.config.layers[i + 1]))
            self.net.append(nn.BatchNorm1d(self.config.layers[i + 1]))
            if self.config.activation == "leaky_relu":
                self.net.append(nn.LeakyReLU())
            elif self.config.activation == "relu":
                self.net.append(nn.ReLU())

        # Letztes Layer ohne BatchNorm und mit Sigmoid-Aktivierung
        output_size = self.config.features
        self.net.append(nn.Linear(self.config.generator_layers[-1], output_size))
        self.net.append(nn.Sigmoid())  # Sigmoid-Aktivierung für das letzte Layer

    def forward(self, x):     
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__() 
        self.config = config
        self.net = nn.Sequential()  

        for i in range(len(self.config.layers) - 1):
            self.net.append(nn.Linear(self.config.layers[i], self.config.layers[i + 1]))
            self.net.append(nn.BatchNorm1d(self.config.layers[i + 1]))
            if self.config.activation == "leaky_relu":
                self.net.append(nn.LeakyReLU())
            elif self.config.activation == "relu":
                self.net.append(nn.ReLU())
    
    def forward(self, x): 
        return self.net(x)