# NOTE this file is only to prepare the conditional inputs for the conditional GAN example

import pandas as pd
import os 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import random

print(os.getcwd())
csvfile = os.path.join(os.getcwd(), "penguins_size.csv")
df = pd.read_csv(csvfile)
df.dropna(inplace=True)

print("Rohes Dataframe")
print(df.head())

print("Info")
print(df.info())

print("Stats")
print(df.describe())

df_count = df.groupby(["species","island","sex"]).count().reset_index()

dotval =  df_count[df_count['sex'] == "."].index
df_count.drop(dotval, inplace=True)
num_combs = df_count["body_mass_g"].sum() #NOTE this is wrong, because it does not count the number of samples per group, but the sum of body mass

df_count["probability"] = df_count["body_mass_g"]/num_combs
print("Groupby")
print(df_count)

df_count.drop(["culmen_length_mm","culmen_depth_mm","flipper_length_mm","body_mass_g"], axis=1, inplace=True)

print("Probabilities")
print(df_count)

encoder = OneHotEncoder()

encoded = encoder.fit_transform(df_count[["species","island","sex"]]).toarray()

print(encoded)


BATCHSIZE = 20


cat = np.vstack(random.choices(encoded, weights=df_count["probability"], k=BATCHSIZE))#[0]

num = torch.rand(BATCHSIZE, 128)

noise_tensor = torch.cat(tensors=(num,torch.from_numpy(cat)), dim=1)

print(noise_tensor)

