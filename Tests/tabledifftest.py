import os
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# NOTE see https://gianluca.ai/table-diffusion/ for reference


# TODO DataPreparation vorbereiten.
# DONE Sinosuidal Embedding einbauen für timestep Identifikation im NN
# TODO WAS mach ich mit kategorischen Daten  ?? <-- One Hote und dann ??? irgendwelche flatteing Geschichten ?
# NOTE Logspace Encoding für OH 
# TODO Training implementieren
# TODO Privacy wie bewerten ??
# TODO Data Generation Routine bauen
# DONE kein Softmax für Diffusion Modell <-- es soll logits ausspucken 
# TODO Logits für Kategorische Daten wärend des Trainings für KL Divergenz (evtl auch für Rekonstruktion?)
# NOTE besser LayerNorm anstatt BatchNorm self.ln = nn.LayerNorm(64) <-- normalisierung über einzelne Zeilen, nicht Batch

# Automatischer Hardware-Switch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple Silicon GPU/Neural Engine (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Nutze NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Standard-CPU")


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
        return x
        #x = torch.softmax(x) # NOTE keine Aktivierungsfunktion im Output Layer ! keine Normalisierung am Ende, only plain logits


class TableDiff():
    def __init__(self):
        pass

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

print(X_test)

# NOTE Training
n_epochs = 100
batch_size = 32
noise_steps = 300
learning_rate = 0.01
embedding_size = 64 # NOTE das Embedding führe ich hier per concatenan an jede Zeile des Mini Batches an 

train_dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=True) 
# Droplast falls nur noch eine oder zwei Zeilen übrig bleiben und keinen vollen Batch mehr ergeben
# Alternativ ohne Drop Last und kein BatchNorm, sondern LayerNorm <-- hier wird eh zeilenweise gearbeitet 
test_dataloader = DataLoader(X_test, batch_size=batch_size, shuffle=True)

#print("dataloader iter" ,next(iter(train_dataloader)))

def calc_beta(t, steps):
    beta_t = (1.0 - np.cos(np.pi*t/steps))/2.0 
    return beta_t

def calc_alpha():
    pass

def show_progress(epoch,t, numeric_loss, categoric_loss):
        #if t % 100 == 0:
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], ', 
                    f'numeric Loss: {numeric_loss.item():.3f}, '
                    f'kl Loss: {categoric_loss.item():.3f}')

def add_time_embeddings(noised_batch, t, embedding_dim=64):
        # this will generate a vector for every t which is unique for every t 
    # the vector is used to give the neural network a glimpse hint which diffusion step is it in.
    # https://neuraloperator.github.io/dev/auto_examples/layers/plot_sinusoidal_embeddings.html
    # https://medium.com/@giovanitavares/sinusoidal-embeddings-how-transformers-interpret-tokens-positions-bd701babb508
    # https://runebook.dev/en/docs/pytorch/generated/torch.nn.embedding  NOTE Alternative zu mathematischem Embedding 
    """
    Inputs:
    - noised_batch: Dein Tensor [20, 13]
    - t: Der aktuelle Zeit-Schritt (Integer, z.B. 150)
    - embedding_dim: Wie "breit" der Zeit-Vektor sein soll (muss gerade Zahl sein)
    """
    
    # 1. Den Sinusoidal-Vektor für einen einzelnen Zeitschritt berechnen
    # (Das ist die mathematische Standard-Formel)
    device = noised_batch.device
    half_dim = embedding_dim // 2
    emb_scale = np.log(10000) / (half_dim - 1)
    emb_scale = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
    
    # t wird mit den verschiedenen Frequenzen multipliziert
    emb = torch.tensor([t], device=device).float() * emb_scale.unsqueeze(0)
    # Sinus und Cosinus mischen für den "Barcode"-Effekt
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # Resultat: [1, 64]

    # 2. Den Vektor "untereinander" kopieren (Expand auf Batch-Größe)
    # .expand liest die Shape von noised_batch ab
    batch_size = noised_batch.shape[0]
    t_emb_expanded = emb.expand(batch_size, -1) # Resultat: [20, 64]

    # 3. Hinten an den Mini-Batch ankleben
    # dim=1 bedeutet "Spalten hinzufügen"
    final_input = torch.cat([noised_batch, t_emb_expanded], dim=1) # Resultat: [20, 77]
    
    return final_input


learning_rate = 0.01
betas = (0.5, 0.999)

diffmodel = Diffusor(input_dim=76, hidden_dim=64, output_size=12)
diffmodel.to(device=device)

optimizer = optim.Adam(diffmodel.parameters(), lr=learning_rate, betas=betas)
mse_loss = nn.MSELoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')
# soft_max = nn.Softmax() # https://dev.to/sabha_naaz_b5fb8be540fc0f/understanding-softmax-and-cross-entropy-in-neural-networks-daa NOTE Softmax and Cross Entropy article
# log_softmax = log_softmax()

print("start training")

for epoch in range(n_epochs):
    for batch in train_dataloader:     
        batch = batch.float().to(device) # daten im dataloader sind double   
        #print(batch, batch.shape)
        #print("Kat", batch[:,:8])
        #print("Num", batch[:,8:])        
        for t in range(1,noise_steps): # Beta wird für t=0 auch null -> gesamtes Rauschen = 0
            """ NOTE wie bei Gianluca verwenden wir hier mixed Type Denoiser
            Es wird nicht klassisch wie bei StableDiffusion für jeden Schritt das Noise berechnet und dann abgezogen
            Es wird einfach die entrauschten Daten berechnet <-- Vorteil man kann direkt den Loss für numerische (MSE)
            und den Loss für kategorische Daten berechnen (KL Divergenz)
            
            NOTE normalerweise wird bei modernen Diffusion die Steps nicht sequentiell sondern zufällig gewählt,
            innerhalb der Grenzen 0-diffusion steps"""
            
            optimizer.zero_grad() 

            beta_t = calc_beta(t, noise_steps)
            noise = torch.randn_like(batch)*np.sqrt(beta_t)
            
            noised_data = batch + noise 
            noised_data_with_embeddings = add_time_embeddings(noised_batch=noised_data,t=t, embedding_dim=embedding_size) # DONE es ist eleganter den Tensor (d.h. alle sinusoidal vektoren n x untereinander) durch eine fkt generieren zu lassen
            
            denoised_data = diffmodel(noised_data_with_embeddings)
            
            denoised_num = denoised_data[:,8:]
            numeric_loss = mse_loss(denoised_num, batch[:,8:])

            # Slicing der Kategorien-Gruppen
            pred_species = denoised_data[:, 0:3] # Art
            pred_island  = denoised_data[:, 3:6] # Insel
            pred_sex     = denoised_data[:, 6:8] # Geschlecht

            # Einzelne KL-Verluste
            loss_species = kl_loss(log_softmax(pred_species, dim=1), batch[:, 0:3]) # TODO NOTE eventually nn.KL_Divloss uses log softmax and not regular softmax 
            loss_island  = kl_loss(log_softmax(pred_island, dim=1),  batch[:, 3:6])
            loss_sex     = kl_loss(log_softmax(pred_sex, dim=1),     batch[:, 6:8])

            # Alles zusammen
            categorical_loss = (loss_species + loss_island + loss_sex) / 3
            total_loss = numeric_loss + categorical_loss # TODO NOTE eventually scale categorical Loss to be competitive to numeric loss ?? 

            total_loss.backward()
            optimizer.step()
            show_progress(epoch,t, numeric_loss, categorical_loss)

def sample(model, dataprep_obj, batch_size, steps, embedding_size=64, export=False, filename="tablediff_export.csv", real_fake_combined=False):
    model.eval()
    
    # Bestimme die Feature-Dimensionen dynamisch aus dem DataPrep-Objekt
    # input_dim des Modells (76) abzüglich embedding_size (64) = 12 Features
    feature_dim = dataprep_obj.total_features 
    
    with torch.no_grad():
        # 1. Start mit reinem Rauschen (Standard-Normalverteilung)
        x = torch.randn(batch_size, feature_dim)
        
        # 2. Rückwärtsprozess (Denoising Loop)
        for t in reversed(range(1, steps)):
            # Zeit-Embedding anhängen (Nutzt deine vorhandene add_time_embeddings Funktion)
            x_with_emb = add_time_embeddings(noised_batch=x, t=t, embedding_dim=embedding_size)
            
            # Modell sagt die bereinigten Daten (x_0) voraus
            x_0_pred = model(x_with_emb)
            
            # Clipping für numerische Stabilität (da MinMaxScaler auf [0, 1] skaliert)
            x_0_pred = torch.clamp(x_0_pred, 0, 1)
            
            # Gewichtung basierend auf deinem Beta-Schedule (DDPM-Stil für Direkt-Denoiser)
            beta_t = calc_beta(t, steps)
            
            # Je kleiner t wird, desto mehr vertrauen wir der x_0 Vorhersage
            x = (1.0 - beta_t) * x_0_pred + beta_t * x
            
            # Optional: Minimales Rauschen hinzufügen, außer im letzten Schritt (Langevin Dynamics)
            if t > 1:
                x += 0.01 * torch.randn_like(x)
                
    # 3. Postprocessing & Inverse Transformation
    # Splitten der generierten Daten in Kategorisch (erste 8 Spalten) und Numerisch (Rest)
    fake_data_cat = x[:, 0:8].cpu().numpy()
    fake_data_num = x[:, 8:].cpu().numpy()
    
    # Zugriff auf die Transformer aus der ColumnTransformer-Instanz
    oh = dataprep_obj.collumn_trans.named_transformers_['cat']
    scaler = dataprep_obj.collumn_trans.named_transformers_['remainder']
    
    # Rücktransformation in echte Werte
    inverse_cat = oh.inverse_transform(fake_data_cat)
    inverse_num = scaler.inverse_transform(fake_data_num)
    
    # Erstelle DataFrames
    categorical_columns = dataprep_obj.categorical_columns
    df_cat = pd.DataFrame(data=inverse_cat, columns=categorical_columns)
    
    # Bestimme die Namen der numerischen Spalten
    num_cols = [col for col in dataprep_obj.df.columns if col not in categorical_columns]
    df_num = pd.DataFrame(data=inverse_num, columns=num_cols)
    
    # Zusammenfügen
    if not real_fake_combined:
        df = pd.concat([df_cat, df_num], axis=1)
    else:
        df_real = dataprep_obj.df.sample(n=batch_size, random_state=42).reset_index(drop=True)
        df_real["source"] = "real"
        df_fake = pd.concat([df_cat, df_num], axis=1)   
        df_fake["source"] = "fake"  
        df = pd.concat([df_real, df_fake], axis=0).reset_index(drop=True)   
        
    if export:        
        df.to_csv(filename, index=False)
        print(f"Daten erfolgreich in {filename} gespeichert!")
    else:
        print("\n--- Generierte Sample-Daten (Head) ---")
        print(df.head())
        
    return df

# Aufruf der Methode nach dem Training:
sampled_df = sample(
    model=diffmodel, 
    dataprep_obj=data, 
    batch_size=200, 
    steps=noise_steps, 
    embedding_size=embedding_size,
    real_fake_combined=True,
    export=True
)