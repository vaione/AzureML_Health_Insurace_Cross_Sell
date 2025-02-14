import argparse
import json
import os
#import azureml.core
from azureml.core import Model
from azureml.core import Run
#from azureml.train.hyperdrive import HyperDriveRun
from shutil import copy2

parser = argparse.ArgumentParser("Register_model")
parser.add_argument('--saved_model', type=str, help='path to saved model file')
args = parser.parse_args()

model_output_dir = './model/'

os.makedirs(model_output_dir, exist_ok=True)
copy2(args.saved_model, model_output_dir)

ws = Run.get_context().experiment.workspace

model = Model.register(workspace=ws, model_name='Health_XGBoost_HD', model_path=model_output_dir)
