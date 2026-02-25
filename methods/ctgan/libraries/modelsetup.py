import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(in_features=noise_dim, out_features=hidden_dim) 
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.l2 = nn.Linear(in_features=hidden_dim, out_features=2*hidden_dim) 
        self.bn2 = nn.BatchNorm1d(2*hidden_dim)
        self.l3 = nn.Linear(in_features=2*hidden_dim, out_features=output_size) 
        #self.relu = nn.LeakyReLU() 
        self.relu = nn.ReLU()         

    def forward(self, x): 
        x = self.l1(x) 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.l3(x)   
        x = torch.sigmoid(x)     
        return x # NOTE Features wurden über MinMax Scaler normalisiert [0,1] <-- Sigmoid sorgt dafür das der Output auch zwischen 0 und 1 ist
    
    def generate_samples(self, n):
        pass

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes= 1):
        super(Discriminator, self).__init__() 
        self.l1 = nn.Linear(in_features=input_size, out_features=hidden_size) 
        self.ln1 = nn.LazyBatchNorm1d(hidden_size)
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(in_features=hidden_size, out_features=num_classes) 
    
    def forward(self, x): 
        x = self.l1(x) 
        x = self.ln1(x)
        x = self.relu(x) 
        x = self.l2(x) 
        x = torch.sigmoid(x)
        return x