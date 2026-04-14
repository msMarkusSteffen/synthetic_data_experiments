import os
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# TODO DataPreparation vorbereiten.
# TODO Sinosuidal Embedding einbauen für timestep Identifikation im NN
# TODO WAS mach ich mit kategorischen Daten  ?? <-- One Hote und dann ??? irgendwelche flatteing Geschichten ?
# NOTE Logspace Encoding für OH 
# TODO Training implementieren
# TODO Privacy wie bewerten ??
# TODO Data Generation Routine bauen


class DataPrep():
    def __init__(self, datafile, categorical_columns, noise_dim, value_filter=["."], ):
        self.categorical_columns = categorical_columns
        self.df = pd.read_csv(datafile)
        self.df.dropna(inplace=True)

        self.noise_dim = noise_dim
        self.full_noise_dim = None

        self.generator_features = len(self.df.columns)-len(self.categorical_columns)
        
        for filter_val in value_filter:
            for col in self.categorical_columns:
                values= self.df[self.df[col] == filter_val].index
                self.df.drop(values, inplace=True)

        self.df_count = self.df.groupby(self.categorical_columns).count().reset_index()

        num_combs = self.df_count.iloc[:,len(self.categorical_columns)+1].sum()
        self.df_count["probability"] = [x/num_combs for x in self.df_count.iloc[:,len(self.categorical_columns)+1]]
        
        self.__init_preprocessing_models()

    def __init_preprocessing_models(self):
        self.encoder_noise  = OneHotEncoder()
        self.collumn_trans  = ColumnTransformer(transformers=[("cat", OneHotEncoder(), self.categorical_columns)],remainder=MinMaxScaler())
        self.encoded_noisecondition_tensor = self.encoder_noise.fit_transform(self.df_count[self.categorical_columns]).toarray() 
        self.full_noise_dim = self.noise_dim + self.encoded_noisecondition_tensor.shape[1]

    def generate_training_test_data(self, boootstrap_multiplier=10, test_size=0.33, random_state=42):
        transformed = self.collumn_trans.fit_transform(self.df)
        #print("Transformed_Train", transformed)

        X = resample(transformed,replace=True,n_samples=boootstrap_multiplier*len(self.df),random_state=random_state) 

        self.total_features = X.shape[1] # Alle Spalten nach der Transformation
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        return X_train, X_test



def sinosuidal_embedding(t, dimensions):
    # this will generate a vector for every t which is unique for every t 
    # the vector is used to give the neural network a glimpse hint which diffusion step is it in.
    # https://neuraloperator.github.io/dev/auto_examples/layers/plot_sinusoidal_embeddings.html
    # https://medium.com/@giovanitavares/sinusoidal-embeddings-how-transformers-interpret-tokens-positions-bd701babb508
    # https://runebook.dev/en/docs/pytorch/generated/torch.nn.embedding  NOTE Alternative zu mathematischem Embedding 


    pass

# NOTE see https://gianluca.ai/table-diffusion/ for reference

class Diffusor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super(Diffusor, self).__init__()
        self.l1 = nn.Linear(in_features=input_dim, out_features=hidden_dim) 
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


def build_noise_layers(shape, steps=1):
    noise_layers = []
    
    for s in range(1,steps+1):
        # beta_s = s/steps
        beta_s = (1.0 - np.cos(np.pi*s/steps))/2.0 
        # NOTE as to mentioned in the upper reference using cos instead of linear noise reduce the forward 
        # process steps by an order of a magnitude (implementing nonlinearity)
        noise_layer = torch.randn(shape) *np.sqrt(beta_s) # NOTE randn for normal distribution
        noise_layers.append(noise_layer)
    return noise_layers 
X= torch.ones(8, requires_grad=True)
steps = 3
print(build_noise_layers(X.shape, steps))
noise_layers = build_noise_layers(X.shape, steps)
"""
for layer in noise_layers:
    X = X+layer

print("noisedX", X)

for layer in noise_layers:
    X = X-layer

print("denoisedX", X)
"""

emb = nn.Embedding(num_embeddings=30, embedding_dim=10)
t = torch.tensor([1,2,3,4,5,6,7,8,9], dtype=torch.long)
#print(emb(t))
#print(t)

csvfile = os.path.join(os.getcwd(), "datasets/penguins_size.csv")
data = DataPrep(categorical_columns=["species","island","sex"], datafile=csvfile,noise_dim=128)
X_train, X_test= data.generate_training_test_data(boootstrap_multiplier=10)

# NOTE Training
n_epochs = 1
batch_size = 5
noise_steps = 300
learning_rate = 0.01


train_dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(X_test, batch_size=batch_size, shuffle=True)

#print("dataloader iter" ,next(iter(train_dataloader)))

def calc_beta(t, steps):
    beta_t = (1.0 - np.cos(np.pi*t/steps))/2.0 
    return beta_t
    

diffmodel = Diffusor(input_dim=128, hidden_dim=64, output_size=12)
criterion = nn.MSELoss()
optimizer = optim.Adam(diffmodel.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    for batch in train_dataloader:
        #print(batch, batch.shape)
        print("Kat", batch[:,:8])
        print("Num", batch[:,8:])
        break
        for t in range(0,noise_steps):
            beta_t = calc_beta(t, noise_steps)
            noise = torch.randn_like(batch)*np.sqrt(beta_t)
            noised_data = batch + noise 
            predicted_noise = diffmodel(noised_data)

