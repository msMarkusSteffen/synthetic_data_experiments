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

# NOTE see https://gianluca.ai/table-diffusion/ for reference


# TODO DataPreparation vorbereiten.
# TODO Sinosuidal Embedding einbauen für timestep Identifikation im NN
# TODO WAS mach ich mit kategorischen Daten  ?? <-- One Hote und dann ??? irgendwelche flatteing Geschichten ?
# NOTE Logspace Encoding für OH 
# TODO Training implementieren
# TODO Privacy wie bewerten ??
# TODO Data Generation Routine bauen
# TODO kein Softmax für Diffusion Modell <-- es soll logits ausspucken 
# TODO Logits für Kategorische Daten wärend des Trainings für KL Divergenz (evtl auch für Rekonstruktion?)

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
        x = torch.softmax(x) 


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

#emb = nn.Embedding(num_embeddings=30, embedding_dim=10)
#t = torch.tensor([1,2,3,4,5,6,7,8,9], dtype=torch.long)
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
    

def sinosuidal_embedding(t, embedding_size):
    embedding_vect = []
    for i in range(0,embedding_size):
        if i%2 == 0:
            embedding_vect.append(np.sin(t/(1000.0*np.exp(2.0*i/embedding_size))))
        else:
            embedding_vect.append(np.cos(t/(1000.0*np.exp(2.0*i/embedding_size))))
    return torch.Tensor(embedding_vect)


def sinosuidal_embedding_torch(t, embedding_size):
    """
    t: Ein Tensor mit den Zeitschritten (Batch_Size,)
    embedding_size: Die gewünschte Breite (z.B. 32)
    """
    # 1. Wir berechnen den "Divisor" Teil der Formel: 10000^(2i/D)
    # Das machen wir nur für die Hälfte der Dimensionen
    half_dim = embedding_size // 2
    
    # Wir nutzen log/exp für numerische Stabilität: 
    # exp(log(10000) * i / half_dim) ist das gleiche wie 10000^(2i/D)
    emb_base = np.log(10000) / (half_dim - 1)
    # Das hier erzeugt die verschiedenen Frequenzen
    div_term = torch.exp(torch.arange(half_dim) * -emb_base).to(t.device)
    
    # 2. t (Batch, 1) multipliziert mit div_term (Half_Dim) 
    # ergibt eine Matrix (Batch, Half_Dim)
    # t[:, None] macht aus [20] -> [20, 1]
    args = t[:, None].float() * div_term[None, :]
    
    # 3. Jetzt Sinus und Cosinus berechnen und zusammenkleben
    # Das ergibt (Batch, embedding_size)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    return embedding

def get_sinusoidal_embeddings(t_batch, embedding_dim):
    """
    t_batch: Tensor der Form (batch_size,) mit Integern von 0 bis max_steps
    embedding_dim: z.B. 32
    """
    device = t_batch.device
    half_dim = embedding_dim // 2
    
    # 1. Erzeuge die exponentiell abfallenden Frequenzen
    # Formel: 10000^(-2i/D)
    emb_base = math.log(10000) / (half_dim - 1)
    frequencies = torch.exp(torch.arange(half_dim, device=device) * -emb_base)
    
    # 2. t_batch (Batch, 1) * frequencies (1, half_dim) -> (Batch, half_dim)
    args = t_batch[:, None].float() * frequencies[None, :]
    
    # 3. Kombiniere Sinus und Cosinus zu (Batch, embedding_dim)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    return embedding

def get_sinusoidal_embeddings(t, embedding_dim):
    """
    t: Tensor der Form (batch_size,) - Die Zeitschritte für den Batch
    embedding_dim: Wie breit soll der Zeit-Vektor sein? (z.B. 32)
    """
    # 1. Berechne den Nenner der Formel (die Frequenzen)
    # Wir nehmen nur die Hälfte der Dimension, da wir sin und cos kombinieren
    half_dim = embedding_dim // 2
    
    # Frequenzen berechnen: 10000^(-(0..2i)/D)
    emb_base = np.log(10000) / (half_dim - 1)
    frequencies = torch.exp(torch.arange(half_dim) * -emb_base).to(t.device)
    
    # 2. t mit Frequenzen multiplizieren
    # t[:, None] macht aus (batch,) -> (batch, 1)
    # frequencies[None, :] macht aus (half_dim,) -> (1, half_dim)
    # Das Ergebnis ist (batch, half_dim)
    args = t[:, None].float() * frequencies[None, :]
    
    # 3. Sinus und Cosinus berechnen und zusammenfügen
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    return embedding
    

diffmodel = Diffusor(input_dim=128, hidden_dim=64, output_size=12)
mse_loss = nn.MSELoss()
kl_loss = nn.KLDivLoss()

optimizer = optim.Adam(diffmodel.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    for batch in train_dataloader:        
        #print(batch, batch.shape)
        print("Kat", batch[:,:8])
        print("Num", batch[:,8:])        
        for t in range(0,noise_steps):
            """ NOTE wie bei Gianluca verwenden wir hier mixed Type Denoiser
            Es wird nicht klassisch wie bei StableDiffusion für jeden Schritt das Noise berechnet und dann abgezogen
            Es wird einfach die entrauschten Daten berechnet <-- Vorteil man kann direkt den Loss für numerische (MSE)
            und den Loss für kategorische Daten berechnen (KL Divergenz)
            
            NOTE normalerweise wird bei modernen Diffusion die Steps nicht sequentiell sondern zufällig gewählt,
            innerhalb der Grenzen 0-diffusion steps"""
            
            beta_t = calc_beta(t, noise_steps)
            noise = torch.randn_like(batch)*np.sqrt(beta_t)
            noised_data = batch + noise 
            denoised_data = diffmodel(noised_data, t)

            denoised_num = denoised_data[:,8:]
            numeric_loss = mse_loss(denoised_num, batch[:,8:])

            # Slicing der Kategorien-Gruppen
            pred_species = denoised_data[:, 0:3] # Art
            pred_island  = denoised_data[:, 3:6] # Insel
            pred_sex     = denoised_data[:, 6:8] # Geschlecht

            # Einzelne KL-Verluste
            loss_species = F.kl_div(F.log_softmax(pred_species, dim=1), target[:, 0:3], reduction='batchmean')
            loss_island  = F.kl_div(F.log_softmax(pred_island, dim=1),  target[:, 3:6], reduction='batchmean')
            loss_sex     = F.kl_div(F.log_softmax(pred_sex, dim=1),     target[:, 6:8], reduction='batchmean')

            # Alles zusammen
            categorical_loss = (loss_species + loss_island + loss_sex) / 3
            total_loss = numeric_loss + categorical_loss
            

def sample(model, batch_size, steps, feature_dim):
    model.eval()
    with torch.no_grad():
        # 1. Start mit reinem Rauschen
        x = torch.randn(batch_size, feature_dim)
        
        for t in reversed(range(1, steps)):
            # Zeit-Tensor für den aktuellen Schritt (alle im Batch haben gleiches t)
            t_tensor = torch.full((batch_size,), t, dtype=torch.long)
            t_emb = sinusoidal_embedding(t_tensor, embedding_size)
            
            # Modell sagt saubere Daten voraus
            x_0_pred = model(torch.cat([x, t_emb], dim=1))
            
            # Clipping/Post-processing (Wichtig für Stabilität!)
            # Da wir wissen, dass unsere Daten skaliert sind (0 bis 1)
            x_0_pred = torch.clamp(x_0_pred, 0, 1)
            
            # Schritt zurück berechnen (Vereinfachte DDPM Logik)
            # Wir mischen den vorhergesagten x_0 mit dem aktuellen x
            alpha_t = get_alpha(t) # Aus deinem Noise-Schedule
            x = (1 - alpha_t) * x_0_pred + alpha_t * x
            
            # Optional: Wieder ein ganz kleines bisschen Rauschen addieren 
            # (Langevin Dynamics), um den Prozess lebendig zu halten
            if t > 1:
                x += 0.01 * torch.randn_like(x)
                
        return x