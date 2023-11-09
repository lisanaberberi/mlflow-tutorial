#!/usr/bin/env python
# coding: utf-8

# ## 1. Checking the environment
# 

import sys
print(sys.executable)


# If Tensorflow is installed
import tensorflow as tf
print(f'Tensorflow version = {tf.__version__}\n')


# 
# ## 2. MLflow Components
# 
# The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models. It provides model lineage (which MLflow Experiment and Run produced the model), model versioning, stage transitions, annotations, and deployment management.
# 
# In this notebook, you use each of the MLflow Model Registry's components to develop and manage a production machine learning application. This notebook covers the following topics:
# * Log model experiments with MLflow
# * Register models with the Model Registry
# * Add model description and version the model stage transitions
# * Search and discover models in the MR
# * Archive and delete models

# ## 2.1 ML Application Example using MLFlow
# ## Load the dataset
# 
# The following cells load a dataset containing weather data and power output information for a wind farm in the United States. The dataset contains wind direction, wind speed, and air temperature features sampled every eight hours (once at 00:00, once at 08:00, and once at 16:00), as well as daily aggregate power output (power), over several years.
# 

import pandas as pd

data_source_url = "https://github.com/dbczumar/model-registry-demo-notebook/raw/master/dataset/windfarm_data.csv"
wind_farm_data = pd.read_csv(data_source_url, index_col=0)


def get_training_data():
  training_data = pd.DataFrame(wind_farm_data["2014-01-01":"2018-01-01"])
  X = training_data.drop(columns="power")
  y = training_data["power"]
  return X, y


def get_validation_data():

  validation_data = pd.DataFrame(wind_farm_data["2018-01-01":"2019-01-01"])
  X = validation_data.drop(columns="power")
  y = validation_data["power"]
  return X, y


def get_weather_and_forecast():

  format_date = lambda pd_date : pd_date.date().strftime("%Y-%m-%d")
  today = pd.Timestamp('today').normalize()
  week_ago = today - pd.Timedelta(days=5)
  week_later = today + pd.Timedelta(days=5)

  past_power_output = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(today)]

  weather_and_forecast = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(week_later)]

  if len(weather_and_forecast) < 10:
    past_power_output = pd.DataFrame(wind_farm_data).iloc[-10:-5]
    weather_and_forecast = pd.DataFrame(wind_farm_data).iloc[-10:] 

  return weather_and_forecast.drop(columns="power"), past_power_output["power"]


# #### 1. Display a sample of the data for reference.

wind_farm_data["2019-01-01":"2019-02-01"]

# ## Train a power forecasting model and track it with MLflow
# 
# The following cells train a neural network to predict power output based on the weather features in the dataset. MLflow is used to track the model's hyperparameters, performance metrics, source code, and artifacts.
# 

# #### 2. Define a power forecasting model using TensorFlow Keras.

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Check for available physical devices
gpus = tf.config.list_physical_devices("GPU")
cpus = tf.config.list_physical_devices("CPU")

if len(gpus) > 0:
    print(gpus)
else:
    print("NO GPUS FOUND! Using only CPUs!")

print(f'\n{cpus}\n')

def train_keras_model(X, y):
  
  model = Sequential()
  model.add(Dense(100, input_shape=(X_train.shape[-1],), activation="relu", name="hidden_layer"))
  model.add(Dense(1))
  model.compile(loss="mse", optimizer="adam")

  history=model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=.2)

  return model, history


# #### MLflow part
# 
# **! Configure IMPORTANT CONSTANTS !:**

#set the environmental vars to allow 'mlflow_user' to track experiments using MLFlow
import os
import getpass

# IMPORTANT CONSTANTS TO DEFINE
# MLFLOW CREDENTIALS (Nginx). PUT REAL ONES!
# for direct API calls via HTTP we need to inject credentials
MLFLOW_TRACKING_USERNAME = input('Enter your username: ')
MLFLOW_TRACKING_PASSWORD =  getpass.getpass()  # inject password by typing manually
# for MLFLow-way we have to set the following environment variables
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD


import subprocess
# Run the git command and capture the output
result = subprocess.run(["git", "config", "--get", "remote.origin.url"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Check if the command was successful
if result.returncode == 0:
    # Get the output of the command
    git_url = result.stdout

    # Write the output to a text file
    with open("mlflow_example/links.txt", "w") as f:
        lines = ["Git Repo URL: ", git_url, "\n", "Dataset URL: ", data_source_url,"\n" ]
        f.writelines(lines)
else:
    # Handle any errors or log them
    error_message = result.stderr
    print(f"Error: {error_message}")


# Remote MLFlow server
MLFLOW_REMOTE_SERVER="https://mlflow.dev.ai4eosc.eu" 
# Name of the experiment (e.g. name of the  code repository)
MLFLOW_EXPERIMENT_NAME="wind_power_forecast_W"
# Name of the model to train. HAS TO BE UNIQUE, Please, DEFINE ONE!
MLFLOW_MODEL_NAME="Wind_Forecast_W"


# Train the model and use MLflow to log and track its parameters, metrics, artifacts, and source code.

import mlflow
import mlflow.keras
import mlflow.tensorflow
from mlflow.data.pandas_dataset import PandasDataset

from mlflow.models.signature import infer_signature
#from mlflow.models.signature import ModelSignature
#from mlflow.types.schema import Schema, ColSpec

X_train, y_train = get_training_data()

#Set the MLflow server and backend and artifact stores
mlflow.set_tracking_uri(MLFLOW_REMOTE_SERVER)

#set an experiment name for all different runs
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

#Log the dataset

dataset: PandasDataset = mlflow.data.from_pandas(wind_farm_data, source=data_source_url)

with mlflow.start_run():

  # Automatically capture the model's parameters, metrics, artifacts,and source code with the `autolog()` function

  mlflow.log_input(dataset, context="training")

    # Log the parameters
  mlflow.log_params({
      "hidden_units": 100,
      "activation": "relu",
      "epochs": 100,
      "batch_size": 64,
      "validation_split": 0.2
  })

  #mlflow.tensorflow.autolog()  
  model, history=train_keras_model(X_train, y_train)
  
  #Log the project info
  mlflow.log_artifact('mlflow_example/MLproject',artifact_path="project-info")
  mlflow.log_artifact('mlflow_example/links.txt',artifact_path="project-info")

    # Log the metrics
  mlflow.log_metrics({
      "train_loss": history.history["loss"][-1],
      "val_loss": history.history["val_loss"][-1]
  })
 
  # Infer the input and output schemas

  # Assuming you have a DataFrame named 'X_train' with the training data
  #input_signature, output_signature = infer_signature(X_train)
  infer_signature(X_train)


 #Save the Keras model as a TensorFlow SavedModel
  model.save("my_keras_model")

  # Log the TensorFlow SavedModel using mlflow.tensorflow.log_model
  mlflow.tensorflow.log_model(model, artifact_path="source-files")

  # Retrieve the run, including dataset information
  run = mlflow.get_run(mlflow.last_active_run().info.run_id)
  run_id = mlflow.active_run().info.run_id
  dataset_info = run.inputs.dataset_inputs[0].dataset
  print(f"Dataset name: {dataset_info.name}")
  print(f"Dataset digest: {dataset_info.digest}")
  print(f"Dataset profile: {dataset_info.profile}")
  print(f"Dataset schema: {dataset_info.schema}")

  #Log the dataset ready for download
  # Save the DataFrame to a CSV file
  data_csv = "windfarm_data.csv"
  wind_farm_data.to_csv(data_csv, index=False)

  # Log the CSV file as an artifact in MLflow
  mlflow.log_artifact(data_csv, artifact_path='source-files/data/dataset')

# Print the run_id
print(F"\nRUN_ID: {run_id} \n")


# ### Register the model with the MLflow Model Registry API
# 
# Now that a forecasting model has been trained and tracked with MLflow, the next step is to register it with the MLflow Model Registry. You can register and manage models using the MLflow UI or the MLflow API .
# 
# The following cells use the API to register your forecasting model, add rich model descriptions, and perform stage transitions. See the documentation for the UI workflow.

# Create a new registered model using the API
# 
# The following cells use the `mlflow.register_model()` function to create a new registered model whose name begins with the string defined in `MLFLOW_MODEL_NAME`. This also creates a new model version (for example, Version 1 of power-forecasting-model).
# +-***************************************************************************

# The default path where the MLflow autologging function stores the model
model_uri = F"runs:/{run_id}/model"
model_details = mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)


import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus


def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=MLFLOW_MODEL_NAME,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)


wait_until_ready(model_details.name, model_details.version)

# ### Add model descriptions

# Add a high-level description to the registered model, including the machine learning problem and dataset.

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
client.update_registered_model(
  name=model_details.name,
  description="This model forecasts the power output of a wind farm based on weather data. The weather data consists of three features: wind speed, wind direction, and air temperature. // user"
)

# Perform a model staging using RestAPI
# First need to check if api is working:
# http://   /api/2.0/mlflow
# 
# restapi tutorial to transition stage: https://mlflow.org/docs/latest/rest-api.html#transition-modelversion-stage
# 
# 

# Let's use requests library to perform API call
import requests
from requests.auth import HTTPBasicAuth

# Get info about the run
auth=HTTPBasicAuth(MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD)
headers = {'content-type': 'application/json'}
params = { 'run_id' : run_id }
url = MLFLOW_REMOTE_SERVER + "/api/2.0/mlflow/runs/get"
response = requests.get(url, auth=auth, headers=headers, params=params)
response.json()

# Transition model to Production using API call
auth=HTTPBasicAuth(MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD)
headers = {'content-type': 'application/json'}
data = {"comment": "Please move this model into production!", 
        "name": MLFLOW_MODEL_NAME, 
        "version": "1", 
        "stage": "Production"}
url = MLFLOW_REMOTE_SERVER + "/api/2.0/mlflow/model-versions/transition-stage"

response = requests.post(url, auth=auth, headers=headers, json=data)
response.json()

# or using the script:

client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production'

)

# Use the `MlflowClient.get_model_version()` function to fetch the model's current stage.

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)

print(F"The current model stage is: '{model_version_details.current_stage}'")

# The MLflow Model Registry allows multiple model versions to share the same stage. When referencing a model by stage, the Model Registry will use the latest model version (the model version with the largest version ID). The MlflowClient.get_latest_versions() function fetches the latest model version for a given stage or set of stages. The following cell uses this function to print the latest version of the power forecasting model that is in the Production stage.
latest_version_info = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
latest_production_version = latest_version_info[0].version

print(F"The latest production version of the model \"{MLFLOW_MODEL_NAME}\" is {latest_production_version}")


# Cite dataset:

# This notebook uses altered data from the National WIND Toolkit dataset provided by NREL, which is publicly available and cited as follows:
# 
# Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. Overview and Meteorological Validation of the Wind Integration National Dataset Toolkit (Technical Report, NREL/TP-5000-61740). Golden, CO: National Renewable Energy Laboratory.
# 
# Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. "The Wind Integration National Dataset (WIND) Toolkit." Applied Energy 151: 355366.
# 
# Lieberman-Cribbin, W., C. Draxl, and A. Clifton. 2014. Guide to Using the WIND Toolkit Validation Code (Technical Report, NREL/TP-5000-62595). Golden, CO: National Renewable Energy Laboratory.
# 
# King, J., A. Clifton, and B.M. Hodge. 2014. Validation of Power Output for the WIND Toolkit (Technical Report, NREL/TP-5D00-61714). Golden, CO: National Renewable Energy Laboratory.
# 
# And notebook reference here: https://docs.databricks.com/_extras/notebooks/source/mlflow/mlflow-model-registry-example.html
