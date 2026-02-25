import yaml

class DatasetConfig:
    def __init__(self, config):
        self.name = config["name"]
        self.path = config["path"]
        self.categorical_columns = config["categorical_columns"]
        self.numeric_columns = config["numeric_columns"]
        self.bootstrap_multiplier = config["bootstrapmultiplier"]

class GeneratorConfig:
    def __init__(self, config):
        self.noise_dim = config["noise_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.layers = config["generator_layers"]
        self.activation = config["activation"]

    def set_outputfeautures(self, num_feautures):
        self.features = num_feautures

class DiscriminatorConfig:
    def __init__(self, config):
        self.hidden_dim = config["hidden_dim"]
        self.layers = config["discriminator_layers"]
        self.activation = config["activation"]

class TrainingConfig:
    def __init__(self, config):
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.lr_gen = config["lr_gen"]
        self.lr_disc = config["lr_disc"]
        self.beta1 = config["beta1"]
        self.use_wasserstein = config["use_wasserstein"]
        self.export_generator = config["export_generator"]
        self.exported_generator_filename = config["exported_generator_filename"]

class DataGenerationConfig:
    def __init__(self, config):
        self.use_existing_model = config["use_existing_model"]
        self.modelfile = config["modelfile"]
        self.export = config["export"]
        self.size = config["size"]
        self.filename = config["filename"]


class CTGANConfig():
    def __init__(self, yamlfile):  
        ctganpresets = None    
        with open(yamlfile, 'r') as file:
            ctganpresets = yaml.safe_load(file)

        # Defining dataset parameters
        self.dataset = DatasetConfig(ctganpresets["dataset"])
        #self.dataset_name = self.ctganpresets["dataset"]["name"]
        #self.dataset_path = self.ctganpresets["dataset"]["path"]
        #self.dataset_categorical_columns = self.ctganpresets["dataset"]["categorical_columns"]
        #self.dataset_numeric_columns = self.ctganpresets["dataset"]["numeric_columns"]
        #self.dataset_bootstrapmultiplier = self.ctganpresets["dataset"]["bootstrapmultiplier"]
        
        # Defining generator parameters
        self.generator = GeneratorConfig(ctganpresets["generator_architecture"])
        self.generator.set_outputfeautures(len(self.dataset.numeric_columns))

        #self.generator_noise_dim = self.ctganpresets["generator_architecture"]["noise_dim"]
        #self.generator_hidden_dim = self.ctganpresets["generator_architecture"]["hidden_dim"]
        #self.generator_layers = self.ctganpresets["generator_architecture"]["generator_layers"]
        #self.generator_activation = self.ctganpresets["generator_architecture"]["activation"]

        # Defining generator parameters
        self.discriminator = DiscriminatorConfig(ctganpresets["discriminator_architecture"])
        #self.discriminator_hidden_dim= self.ctganpresets["discriminator_architecture"]["hidden_dim"]
        #self.discriminator_layers = self.ctganpresets["discriminator_architecture"]["discriminator_layers"]
        #self.discriminator_activation = self.ctganpresets["discriminator_architecture"]["activation"]

        # Defining training parameters
        self.training = TrainingConfig(ctganpresets["training"])
        #self.training_epochs = self.ctganpresets["training"]["epochs"]
        #self.training_batch_size = self.ctganpresets["training"]["batch_size"]
        #self.training_lr_gen = self.ctganpresets["training"]["lr_gen"]
        #self.training_lr_disc = self.ctganpresets["training"]["lr_disc"]
        #self.training_beta1 = self.ctganpresets["training"]["beta1"]
        #self.training_use_wasserstein = self.ctganpresets["training"]["use_wasserstein"]
        #self.training_export_generator = self.ctganpresets["training"]["export_generator"]
        #self.training_exported_generator_filename = self.ctganpresets["training"]["exported_generator_filename"]

        # Defining data generation parameters
        self.datagen = DataGenerationConfig(ctganpresets["datageneration"])
        #self.datagen_use_existing_model = self.ctganpresets["datageneration"]["use_existing_model"]
        #self.datagen_modelfile = self.ctganpresets["datageneration"]["modelfile"]
        #self.datagen_export = self.ctganpresets["datageneration"]["export"]
        #self.datagen_size = self.ctganpresets["datageneration"]["size"]
        #self.datagen_filename = self.ctganpresets["datageneration"]["filename"]