import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.utils import resample # NOTE for bootstrapping

# Data Loading and Preparation
data = load_iris()
#df = pd.DataFrame(data=data.data, columns=data.feature_names) 
#df['species'] = data.target_names[data.target]
#df["species"]=df["species"].astype("category")

#print(df.info())
#print(df.head())
#print(df['species'].unique())

sc = StandardScaler()
onehot  = OneHotEncoder()


encoded_species = onehot.fit_transform(data.target.reshape(-1,1)).toarray()
#print(oh.fit_transform(data.target))
X = data.data
multiplicator = 10
size = X.shape[0]*multiplicator

X = sc.fit_transform(X=X)
X = np.append(X, encoded_species, axis=1) 
X = resample(X,replace=True,n_samples=size,random_state=42) # Bootstrapping, replace= True (mit zurücklegen)

# Train Test Split
X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)#, stratify=True)

X_orig_train = torch.from_numpy(X_train).float() 
Y_orig_train = torch.ones(X_orig_train.shape[0], 1) 

X_orig_test = torch.from_numpy(X_test).float() 
Y_orig_test = torch.ones(X_orig_test.shape[0], 1) 

# Model Definitions

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_size):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(in_features=noise_dim, out_features=hidden_dim) 
        self.relu = nn.ReLU() 
        self.l2 = nn.Linear(in_features=hidden_dim, out_features=output_size) 

    def forward(self, x): 
        l1_out = self.l1(x) 
        relu_out = self.relu(l1_out) 
        l2_out = self.l2(relu_out) 
        return torch.tanh(l2_out) # NOTE Features wurden über standard Scaler normalisiert [-1,1] <-- tanh sorgt dafür das der Output auch zwischen -1 und 1 ist
    
    def generate_samples(self, n):
        pass

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes= 1):
        super(Discriminator, self).__init__() 
        self.l1 = nn.Linear(in_features=input_size, out_features=hidden_size) 
        self.relu = nn.LeakyReLU()
        self.l2 = nn.Linear(in_features=hidden_size, out_features=num_classes) 
    
    def forward(self, x): 
        l1_out = self.l1(x) 
        relu_out = self.relu(l1_out) 
        l2_out = self.l2(relu_out) 
        l2_out = torch.sigmoid(l2_out)
        return l2_out 

# Training Initialization
NUM_EPOCHS = 2000
#BATCH_SIZE = 256
learning_rate = 0.001
noise_dim = 128
iris_features = 7


generator = Generator(noise_dim,64, iris_features)
discriminator=Discriminator(iris_features,64)

#criterion = nn.BCEWithLogitsLoss() # NOTE passt sonst nicht mit Sgimoid beim output layer 
criterion = nn.BCELoss() 
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)) 
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# print(X_orig_train.shape[0])

# Training
for epoch in range(NUM_EPOCHS):
    #for i in range(0, X_orig_train.shape[0], BATCH_SIZE):
    
    #noise_vect = torch.rand(X_orig_train.shape[0], noise_dim)
    noise_vect = torch.rand(X_orig_train.shape[0], noise_dim)
    #print(noise_vect)
    #print(X_orig_test[:i])
    

    # Forward Propagation Discriminator
    # NOTE zuerst werden die echten Daten trainiert
    discriminator_optimizer.zero_grad()
    real_pred_labels = discriminator(X_orig_train)
    #print(real_pred_labels.shape)
    
    real_loss = criterion(real_pred_labels,Y_orig_train)
    real_loss.backward()

    # NOTE jetzt bringen wir dem Discriminator die Fakedaten bei (ohne den Generator zu trainieren)
    # hier kein zero_grad() für den Discriminator da es so ähnlich ist wie wenn man die Fake und echten Daten gesammelt übergibt
    # der Gradient muss hier addieren
    fake_gen_data = generator(noise_vect)
    fake_targets = torch.zeros(X_orig_train.shape[0], 1)
    fake_outputs = discriminator(fake_gen_data.detach())    # Das Detach bedeutet der Weg zurück zum Generator wird abgeschnitten, aber der Disc. wird trainiert
    fake_loss = criterion(fake_outputs,fake_targets )
    fake_loss.backward()
    discriminator_optimizer.step()

    # NOTE hier trainieren wir den generator
    generator_optimizer.zero_grad()
    fake_targets = torch.ones(X_orig_train.shape[0], 1)
    fake_outputs = discriminator(fake_gen_data)
    gen_loss = criterion(fake_outputs, fake_targets)
    gen_loss.backward()
    generator_optimizer.step()


    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]', 
                    f'Discriminator Loss: {real_loss.item() + fake_loss.item():.3f}, '
                    f'Generator Loss: {gen_loss.item():.3f}')



# NOTE Wasserstein Metrik
# https://forkxz.github.io/blog/2024/Wasserstein/
"""
def torch_wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    # Ensure that the input tensors are batched
    assert u_values.dim() == 2 and v_values.dim() == 2, "Input tensors must be 2-dimensional (batch_size, num_values)"

    batch_size, u_size = u_values.shape
    _, v_size = v_values.shape

    # Sort the values
    u_sorter = torch.argsort(u_values, dim=1)
    v_sorter = torch.argsort(v_values, dim=1)

    # Concatenate and sort all values for each batch
    all_values = torch.cat((u_values, v_values), dim=1)
    all_values, _ = torch.sort(all_values, dim=1)
    # Compute differences between successive values
    deltas = torch.diff(all_values, dim=1)

    # Get the respective positions of the values of u and v among the values of both distributions
    all_continue = all_values[:, :-1].contiguous()
    u_cdf_indices = torch.searchsorted(u_values.gather(1, u_sorter).contiguous(), all_continue, right=True)
    v_cdf_indices = torch.searchsorted(v_values.gather(1, v_sorter).contiguous(), all_continue, right=True)

    # Calculate the CDFs of u and v using their weights, if specified
    if u_weights is None:
        u_cdf = u_cdf_indices.float() / u_size
    else:
        u_sorted_cumweights = torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(u_weights.gather(1, u_sorter), dim=1)), dim=1)
        u_cdf = u_sorted_cumweights.gather(1, u_cdf_indices) / u_sorted_cumweights[:, -1].unsqueeze(1)

    if v_weights is None:
        v_cdf = v_cdf_indices.float() / v_size
    else:
        v_sorted_cumweights = torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(v_weights.gather(1, v_sorter), dim=1)), dim=1)
        v_cdf = v_sorted_cumweights.gather(1, v_cdf_indices) / v_sorted_cumweights[:, -1].unsqueeze(1)

    return torch.sum(torch.abs(u_cdf - v_cdf) * deltas, dim=1)
"""