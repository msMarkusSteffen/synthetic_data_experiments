import os
from libraries.config import CTGANConfig
#from libraries.dataprep import DataPrep
from libraries.modelsetup import Generator#, Discriminator

from enum import Enum

#from ..utils import logger


class Status(Enum):
    INIT = 0
    YAML = 1
    READY = 2
    TRAIN = 3
    DONE = 4
    DATAGEN = 5
    TEST = 6

class Butler():
    def __init__(self, loglevel="info"):        
        self.status = Status.INIT
        self.loglevel = loglevel

    #@log_execution
    def load_config(self, yamlfile):
        try:
            self.config = CTGANConfig(yamlfile)
            print(self.config.dataset.categorical_columns)
            self.status = Status.YAML
        except Exception as e:
            #logger.error(f"Failed to process data: {e}")
            raise 

    #@log_execution
    def __prepare_training(self):
        if self.status == Status.YAML:
            self.generator = Generator(self.config.generator)
            pass
            self.status = Status.READY
        else: 
            raise # Fehler falls nicht im Yaml status

    def train(self):
        self.__prepare_training()
        if self.status == Status.READY:
            pass

    def generate_data(self):
        if self.status == Status.YAML:
            pass
        elif self.status == Status.DONE:
            pass

    def test(self):
        pass

if __name__ == "__main__":
    yamlfile = os.path.join(os.getcwd(), "model_presets/ctgan_covid_sars2.yaml")
    ctbutler = Butler()
    ctbutler.load_config(yamlfile)
    ctbutler.train() 
    ctbutler.generate_data()


