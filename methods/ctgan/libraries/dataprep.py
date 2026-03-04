class DataPrep():
    def __init__(self, config)#datafile, categorical_columns, noise_dim, value_filter=["."], ):
        self,config = config
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
    

    def gen_noise_tensor(self, batch_size):
        cat = np.vstack(random.choices(self.encoded_noisecondition_tensor , weights=self.df_count["probability"], k=batch_size))#[0]
        num = torch.rand(batch_size, self.noise_dim)
        noise_tensor = torch.cat(tensors=(num,torch.from_numpy(c0at)), dim=1)
        return noise_tensor, cat
        
    def rate_model(self, real_data, generated_data):
        wass_metric = WassersteinMetric()
        distance = wass_metric.forward(real_data, generated_data)
        return distance