import yaml
import sys

#inFile = sys.argv[1]
inFile = 'C://Users//Elisa//progetti assegno di ricerca//EPILESSIA//Drug-Resistant-Epilepsy-DRE-Prediction-Models//config.yaml'

with open(inFile, "r") as f:
    config = yaml.safe_load(f)

edf_folder = config['edf_folder']
excel_path = config['excel_path']
target_freq = config['target_freq']
min_len = config['min_len']
num_classes = config['num_classes']
num_epochs = config['num_epochs']