import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "datasets", "generated", "penguins_fake_real.CSV")

df = pd.read_csv(csv_path)
df.drop(["Unnamed: 0"], inplace=True, axis=1)
print(df.head())

#  Plot correlation matrices
# Compute the correlation matrix
#def plot_corr_mat(dataframes, outputfile):
#    d = dataframes
#
#    # Set up the matplotlib figure
#    f, ax = plt.subplots(figsize=(11, 9))
#    for i in range(0,len(d)):
#        corr = d[i].corr()  
#    
#        # Generate a custom diverging colormap
#        cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
#        # Draw the heatmap with the mask and correct aspect ratio
#        corr_plot = sns.heatmap(corr,cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
#                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
#    f = corr_plot.get_figure()
#    f.savefig(outputfile)

def plot_corr_mat(dataframe, outputfile):
    # Set up the matplotlib figure (wichtig: Jedes Mal eine neue Figure!)
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Korrelation direkt vom übergebenen DataFrame berechnen
    corr = dataframe.corr()  
    
    # Custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Heatmap zeichnen mit fixen Grenzen von -1 bis 1 und echten Zahlen (annot=True)
    sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f",
                cbar_kws={"shrink": .5}, ax=ax)
    
    # Titel hinzufügen, damit man sieht, welcher Plot es ist
    #plt.title(os.path.basename(outputfile).replace(".png", "").upper(), fontsize=14)
    
    plt.tight_layout()
    plt.savefig(outputfile)
    plt.close() #

df_fake = df[df['source'] == "fake"].select_dtypes(exclude=['object'])
df_real = df[df['source'] == "real"].select_dtypes(exclude=['object'])

plot_corr_mat(df_fake, "corr_map_fake.png")
plot_corr_mat(df_real, "corr_map_real.png")

# Plot Scree Plots
components = 4
scaler_fake = StandardScaler()
scaled_fake = scaler_fake.fit_transform(df_fake)
pca_fake = PCA(n_components=components)
pca_fake_fit = pca_fake.fit_transform(scaled_fake)
PC_fake_values = np.arange(pca_fake.n_components_) + 1

scaler_real = StandardScaler()
scaled_real = scaler_real.fit_transform(df_real)
pca_real = PCA(n_components=components)
pca_real_fit = pca_real.fit_transform(scaled_real)
PC_real_values = np.arange(pca_real.n_components_) + 1

plt.plot(PC_fake_values, pca_fake.explained_variance_ratio_, 'o-', linewidth=2, color='red', label="synthetic")
plt.plot(PC_real_values, pca_real.explained_variance_ratio_, 'o-', linewidth=2, color='blue', label ="real")
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.legend()
plt.savefig("plots/screeplot.png")
plt.close()


#scree_plot(df_fake, components=4, outputfile="fake_scree.png")
#scree_plot(df_real, components=4, outputfile="real_scree.png")

# Show Loadings

# Ladewerte extrahieren
loadings_real = pca_real.components_.T * np.sqrt(pca_real.explained_variance_)
loadings_fake = pca_fake.components_.T * np.sqrt(pca_fake.explained_variance_)

# DataFrames für Ladewerte erstellen
loadings_real_df = pd.DataFrame(loadings_real, index=df_real.columns, columns=[f'PC{i+1}' for i in range(loadings_real.shape[1])])
loadings_fake_df = pd.DataFrame(loadings_fake, index=df_fake.columns, columns=[f'PC{i+1}' for i in range(loadings_fake.shape[1])])

# Zusammenführen der Daten für den Plot
loadings_df = loadings_real_df[['PC1']].merge(loadings_fake_df[['PC1']], left_index=True, right_index=True, suffixes=('_real', '_fake'))

# Barchart erstellen
loadings_df.reset_index(inplace=True)
loadings_df.rename(columns={'index': 'Features'}, inplace=True)

# In langes Format umwandeln für Seaborn
loadings_melted = loadings_df.melt(id_vars='Features', value_vars=['PC1_real', 'PC1_fake'], 
                                     var_name='Dataset', value_name='Loading')

# Plot mit Seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='Features', y='Loading', hue='Dataset', data=loadings_melted, palette='muted')
plt.title('Feature Importance for the first principal component')
plt.xlabel('Features')
plt.ylabel('Loadings')
plt.legend(title='dataset')
#plt.xticks(rotation=90)  # Drehen der x-Achsen-Beschriftungen
plt.savefig("plots/loadings.png")
plt.tight_layout()
plt.close()

# t-SNE Plots
sc_tse = StandardScaler()
tsne = TSNE(n_components=2)

x = df.drop(["species", "sex", "island", "source"],axis=1)
y = df["source"]

x_scaled = sc_tse.fit_transform(x)
x_folded = tsne.fit_transform(x_scaled)
print(x_folded)
df_tsne = pd.DataFrame({'tsne_1': x_folded[:,0], 'tsne_2': x_folded[:,1], 'label': y})

plt.figure(figsize=(12, 8))
tse_plot = sns.scatterplot(data=df_tsne, x="tsne_1", y="tsne_2", hue=y)
plt.savefig("plots/tSNE.png")
plt.close()

# Trainloop Plots


# Wasserstein Divergence (im Testloop ?)




