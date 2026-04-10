import torch 
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


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


# NOTE see https://gianluca.ai/table-diffusion/ for reference

class Diffusor(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_size):
        super(Diffusor, self).__init__()
        

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

for layer in noise_layers:
    X = X+layer

print("noisedX", X)

for layer in noise_layers:
    X = X-layer

print("denoisedX", X)


