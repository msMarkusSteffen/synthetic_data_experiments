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

# Automatischer Hardware-Switch
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Nutze Device: {device}")

class DataPrep():
    def __init__(self, datafile, categorical_columns, noise_dim, value_filter=["."]):
        self.categorical_columns = categorical_columns
        self.df = pd.read_csv(datafile)
        self.df.dropna(inplace=True)

        self.noise_dim = noise_dim
        self.full_noise_dim = None

        self.generator_features = len(self.df.columns) - len(self.categorical_columns)
        
        for filter_val in value_filter:
            for col in self.categorical_columns:
                values = self.df[self.df[col] == filter_val].index
                self.df.drop(values, inplace=True)

        self.df_count = self.df.groupby(self.categorical_columns).count().reset_index()

        num_combs = self.df_count.iloc[:, len(self.categorical_columns) + 1].sum()
        self.df_count["probability"] = [x / num_combs for x in self.df_count.iloc[:, len(self.categorical_columns) + 1]]
        
        self.__init_preprocessing_models()

    def __init_preprocessing_models(self):
        self.encoder_noise = OneHotEncoder()
        self.collumn_trans = ColumnTransformer(transformers=[("cat", OneHotEncoder(), self.categorical_columns)], remainder=MinMaxScaler())
        self.encoded_noisecondition_tensor = self.encoder_noise.fit_transform(self.df_count[self.categorical_columns]).toarray() 
        self.full_noise_dim = self.noise_dim + self.encoded_noisecondition_tensor.shape[1]

    def generate_training_test_data(self, bootstrap_multiplier=10, test_size=0.33, random_state=42):
        transformed = self.collumn_trans.fit_transform(self.df)
        X = resample(transformed, replace=True, n_samples=bootstrap_multiplier * len(self.df), random_state=random_state) 
        self.total_features = X.shape[1]
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        return X_train, X_test

class SimpleDiffusor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super(SimpleDiffusor, self).__init__()
        self.l1 = nn.Linear(in_features=input_dim, out_features=hidden_dim) 
        self.ln1 = nn.LayerNorm(hidden_dim)  # LayerNorm statt BatchNorm
        self.l2 = nn.Linear(in_features=hidden_dim, out_features=2 * hidden_dim) 
        self.ln2 = nn.LayerNorm(2 * hidden_dim)
        self.l3 = nn.Linear(in_features=2 * hidden_dim, out_features=output_size) 
        self.relu = nn.ReLU()         

    def forward(self, x): 
        x = self.relu(self.ln1(self.l1(x)))
        x = self.relu(self.ln2(self.l2(x)))
        x = self.l3(x)   
        return x
    

class ComplexDiffusor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super(ComplexDiffusor, self).__init__()
        
        # Schicht 1: Eingang zu erster großer Schicht
        self.l1 = nn.Linear(in_features=input_dim, out_features=hidden_dim) 
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Schicht 2: Noch breiter werden
        self.l2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim * 2) 
        self.ln2 = nn.LayerNorm(hidden_dim * 2)
        
        # Schicht 3: Die Breite halten
        self.l3 = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim * 2) 
        self.ln3 = nn.LayerNorm(hidden_dim * 2)
        
        # Schicht 4: Wieder langsam verkleinern
        self.l4 = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim) 
        self.ln4 = nn.LayerNorm(hidden_dim)
        
        # Schicht 5: Ausgabe-Layer (plain Logits/Werte)
        self.l5 = nn.Linear(in_features=hidden_dim, out_features=output_size) 
        
        # SiLU (Swish) ist die Standard-Aktivierungsfunktion für Diffusion
        self.act = nn.SiLU()         

    def forward(self, x): 
        x = self.act(self.ln1(self.l1(x)))
        x = self.act(self.ln2(self.l2(x)))
        x = self.act(self.ln3(self.l3(x)))
        x = self.act(self.ln4(self.l4(x)))
        x = self.l5(x)   
        return x

# --- Diffusion Mathe Setup (Precomputed für Stabilität) ---
noise_steps = 300
# Cosine Schedule Generierung
def get_cosine_schedule(steps):
    s = np.arange(steps + 1)
    alphas_bar = np.cos((s / steps + 0.008) / (1 + 0.008) * np.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 0.0001, 0.9999)

betas = torch.tensor(get_cosine_schedule(noise_steps), dtype=torch.float32, device=device)
alphas = 1.0 - betas
alphas_hat = torch.cumprod(alphas, dim=0)

def add_time_embeddings(noised_batch, t_tensor, embedding_dim=64):
    device = noised_batch.device
    batch_size = noised_batch.shape[0]
    half_dim = embedding_dim // 2
    emb_scale = np.log(10000) / (half_dim - 1)
    emb_scale = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
    
    # t_tensor ist nun ein Tensor der Form [batch_size]
    emb = t_tensor.float().unsqueeze(1) * emb_scale.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # [batch_size, embedding_dim]

    final_input = torch.cat([noised_batch, emb], dim=1)
    return final_input

# --- Daten laden ---
csvfile = os.path.join(os.getcwd(), "datasets/penguins_size.csv")
# Falls lokal nicht vorhanden, Pfad anpassen!
if not os.path.exists(csvfile):
    # Fallback/Erstellung eines Dummy-Pfads oder Erwartung des Users
    pass

data = DataPrep(categorical_columns=["species", "island", "sex"], datafile=csvfile, noise_dim=128)
X_train, X_test = data.generate_training_test_data(bootstrap_multiplier=10)

# Hyperparameter
n_epochs = 2000
batch_size = 64
learning_rate = 0.001
embedding_size = 64 
mixed_type = False # Schaltet auf conditional um wenn false (bei conditional muss diffusion Prozess nur numerische features vorhersagen)

train_dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=False)

diffmodel = ComplexDiffusor(input_dim=data.total_features + embedding_size, hidden_dim=128, output_size=data.total_features)
diffmodel.to(device=device)

optimizer = optim.Adam(diffmodel.parameters(), lr=learning_rate, betas=(0.9, 0.999))
mse_loss = nn.MSELoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')

print("Starte optimiertes Training...")

for epoch in range(n_epochs):
    epoch_loss = 0
    for batch in train_dataloader:     
        batch = batch.float().to(device)
        current_batch_size = batch.shape[0]
        
        optimizer.zero_grad() 

        # 1. Wähle für JEDE Zeile im Batch ein ZUFÄLLIGES t
        t = torch.randint(1, noise_steps, (current_batch_size,), device=device)
        
        # 2. Mathematisch korrekte Vorwärts-Diffusion (Formel für x_t)
        alpha_hat_t = alphas_hat[t].unsqueeze(1)
        noise = torch.randn_like(batch)
        noised_data = torch.sqrt(alpha_hat_t) * batch + torch.sqrt(1.0 - alpha_hat_t) * noise
        
        # 3. Time Embeddings anhängen
        noised_data_with_embeddings = add_time_embeddings(noised_batch=noised_data, t_tensor=t, embedding_dim=embedding_size)
        
        # 4. Vorhersage des Modells (Direkte x_0 Rekonstruktion)
        denoised_data = diffmodel(noised_data_with_embeddings)
        
        # 5. Loss-Berechnung (Immer gegen das saubere Original 'batch')
        denoised_num = denoised_data[:, 8:]
        numeric_loss = mse_loss(denoised_num, batch[:, 8:])

        pred_species = denoised_data[:, 0:3]
        pred_island  = denoised_data[:, 3:6]
        pred_sex     = denoised_data[:, 6:8]

        loss_species = kl_loss(log_softmax(pred_species, dim=1), batch[:, 0:3])
        loss_island  = kl_loss(log_softmax(pred_island, dim=1),  batch[:, 3:6])
        loss_sex     = kl_loss(log_softmax(pred_sex, dim=1),     batch[:, 6:8])

        categorical_loss = (loss_species + loss_island + loss_sex) / 3
        
        total_loss = 5.0 * numeric_loss + categorical_loss # Eventuell Gewichtung erhöhen für numeric Loss, sonst kat zu groß

        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}] - Avg Loss: {epoch_loss/len(train_dataloader):.4f}")


def sample(model, dataprep_obj, batch_size, steps, embedding_size=64, export=False, filename="tablediff_export_randomnoise.csv", real_fake_combined=False):
    model.eval()
    feature_dim = dataprep_obj.total_features 
    
    with torch.no_grad():
        # 1. Start mit reinem Rauschen
        x = torch.randn(batch_size, feature_dim, device=device)
        
        # 2. Rückwärtsprozess
        for t in reversed(range(1, steps)):
            # Erstelle einen Tensor gefüllt mit dem aktuellen Schritt t für den gesamten Batch
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_with_emb = add_time_embeddings(noised_batch=x, t_tensor=t_tensor, embedding_dim=embedding_size)
            
            # Modell sagt das saubere x_0 voraus
            x_0_pred = model(x_with_emb)
            
            # KEIN CLAMP HIER IN DER SCHLEIFE! Erlaubt dem Rauschen natürliche Varianz.
            
            # Korrekte DDPM Sampler-Formel für Direct-x0-Prediction:
            beta_t = betas[t].unsqueeze(0)
            alpha_t = alphas[t].unsqueeze(0)
            alpha_hat_t = alphas_hat[t].unsqueeze(0)
            alpha_hat_t_minus_1 = alphas_hat[t-1].unsqueeze(0) if t > 1 else torch.ones_like(alpha_hat_t)
            
            # Richtung x_0 Schritt berechnen
            pred_dir = torch.sqrt(alpha_hat_t_minus_1) * beta_t / (1.0 - alpha_hat_t)
            x_0_weight = torch.sqrt(alpha_t) * (1.0 - alpha_hat_t_minus_1) / (1.0 - alpha_hat_t)
            
            x = x_0_weight * x + pred_dir * x_0_pred
            
            # Langevin Noise hinzufügen
            if t > 1:
                noise = torch.randn_like(x)
                # Posterior Varianz
                sigma_t = torch.sqrt((1.0 - alpha_hat_t_minus_1) / (1.0 - alpha_hat_t) * beta_t)
                x += sigma_t * noise
                
    # Nach dem Sampling: Konvertiere zu CPU und wende JETZT das schonende Clamping an
    x = x.cpu()
    # Kategorische Logits (Spalten 0-8) unberührt lassen, Numerische Spalten (8+) auf [0,1] begrenzen
    x_numeric_clamped = torch.clamp(x[:, 8:], 0.0, 1.0)
    
    fake_data_cat = x[:, 0:8].numpy()
    fake_data_num = x_numeric_clamped.numpy()
    
    # Inverse Transformation
    oh = dataprep_obj.collumn_trans.named_transformers_['cat']
    scaler = dataprep_obj.collumn_trans.named_transformers_['remainder']
    
    inverse_cat = oh.inverse_transform(fake_data_cat)
    inverse_num = scaler.inverse_transform(fake_data_num)
    
    categorical_columns = dataprep_obj.categorical_columns
    df_cat = pd.DataFrame(data=inverse_cat, columns=categorical_columns)
    
    num_cols = [col for col in dataprep_obj.df.columns if col not in categorical_columns]
    df_num = pd.DataFrame(data=inverse_num, columns=num_cols)
    
    if not real_fake_combined:
        df = pd.concat([df_cat, df_num], axis=1)
    else:
        df_real = dataprep_obj.df.sample(n=min(batch_size, len(dataprep_obj.df)), random_state=42).reset_index(drop=True)
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

# Aufruf nach Anpassung
sampled_df = sample(
    model=diffmodel, 
    dataprep_obj=data, 
    batch_size=200, 
    steps=noise_steps, 
    embedding_size=embedding_size,
    real_fake_combined=True,
    export=True
)