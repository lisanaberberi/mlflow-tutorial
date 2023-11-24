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
# The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the 
# full lifecycle of MLflow Models. It provides model lineage (which MLflow Experiment and Run produced the model), 
# model versioning, stage transitions, annotations, and deployment management.
# 
# In this python script-part 1, you use each of the MLflow Model Registry's components to develop and manage a production ML application. 
# This script covers the following topics:
# * Log model experiments with MLflow
# * Register models with the Model Registry
# * Add model description and version the model stage transitions
# * Add tags and alias to models and models versions

# ## 2.1 ML Application Example using MLFlow
# ## Load the dataset
# 
# The following cells load a dataset containing weather data and power output information for a wind farm in the United States. 
# The dataset contains wind direction, wind speed, and air temperature features sampled every eight hours 
# (once at 00:00, once at 08:00, and once at 16:00), as well as daily aggregate power output (power), over several years.
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
# The following cells train a neural network to predict power output based on the weather features in the dataset. 
# MLflow is used to track the model's hyperparameters, performance metrics, source code, and artifacts.
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
        f.close()
else:
    # Handle any errors or log them
    error_message = result.stderr
    print(f"Error: {error_message}")

import contextlib
# Function to capture the model summary in a file
def log_model_summary(model, summary_file):
  with open(summary_file, 'w') as f, contextlib.redirect_stdout(f):
      model.summary()
  return summary_file

# Remote MLFlow server
MLFLOW_REMOTE_SERVER="https://mlflow.dev.ai4eosc.eu" 

#Set the MLflow server and backend and artifact stores
mlflow.set_tracking_uri(MLFLOW_REMOTE_SERVER)

# Name of the experiment (e.g. name of the  code repository)
MLFLOW_EXPERIMENT_NAME="wind_power_forecast_W"
# Name of the model to train. HAS TO BE UNIQUE, Please, DEFINE ONE!
MLFLOW_MODEL_NAME="wind-forecast-seq-model-v3.0"


# Train the model and use MLflow to log and track its parameters, metrics, artifacts, and source code.

import mlflow
import mlflow.keras
import mlflow.tensorflow
from mlflow.data.pandas_dataset import PandasDataset

X_train, y_train = get_training_data()

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
  from mlflow.models import ModelSignature
  from mlflow.types.schema import Schema, TensorSpec
  import mlflow.types as types
  import numpy as np

  # Assuming X_train is your input data (Pandas DataFrame or NumPy array)
  input_columns = X_train.columns.tolist()
  output_columns = y_train
  print("input col", input_columns)
  print("\n output col", output_columns)
  # Create a model signature
  input_columns = X_train.columns.tolist()
  input_shape = (-1, len(input_columns))  # Shape based on the number of input features
  input_schema = Schema([TensorSpec(np.dtype(np.float64), input_shape, name=feature) for feature in input_columns])
  output_shape = (-1, len(output_columns)) 

  output_schema = Schema([TensorSpec(np.dtype(np.float64), output_shape, name="power")])

  signature = ModelSignature(inputs=input_schema, outputs=output_schema)

  print("\nsignature", signature)
  #Save the Keras model as a TensorFlow SavedModel locally in your machine
  #model.save("my_keras_model")

  # Log the TensorFlow using mlflow.tensorflow.log_model
  mlflow.tensorflow.log_model(model, artifact_path='artifacts', signature=signature)

  # Set the name of the summary file
  summary_file = "model_summary.txt"
  # Log the model summary to the file
  log_model_summary(model, summary_file)
  mlflow.log_artifact(summary_file, artifact_path='model-summary')

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
print(f"\nRUN_ID: {run_id} \n")


# ### REGISTER MODEL to MODEL REGISTRY
result = mlflow.register_model(
    f"runs:/{run_id}/artifacts/", MLFLOW_MODEL_NAME
)

## Fetch the latest model from the Model Registry
from mlflow.tracking.client import MlflowClient
client = MlflowClient()

#check the latest version of the model
model_version_infos = client.search_model_versions(F"name = '{MLFLOW_MODEL_NAME}'")
last_model_version = max([model_version_info.version for model_version_info in model_version_infos])


# set tags, alias, update and delete them
# create "champion" alias for version 1 of model "wind-forecast-seq-model"
client.set_registered_model_alias(MLFLOW_MODEL_NAME, "champion", last_model_version)

# reassign the "Champion" alias to version 5
#client.set_registered_model_alias(MLFLOW_MODEL_NAME, "Champion", '5')

# get a model version by alias
print(f"\n Model version alias: ",client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "champion"))

# delete the alias
#client.delete_registered_model_alias(MLFLOW_MODEL_NAME, "Champion")

# Set registered model tag
client.set_registered_model_tag(MLFLOW_MODEL_NAME, "task", "classification")
client.set_registered_model_tag(MLFLOW_MODEL_NAME, "author", "lisana.berberi@kit.edu")
client.set_registered_model_tag(MLFLOW_MODEL_NAME, "framework", "tensorflow")
# Delete registered model tag
#client.delete_registered_model_tag(MLFLOW_MODEL_NAME, "task")

# Set model version tag
client.set_model_version_tag(MLFLOW_MODEL_NAME, last_model_version, "validation_status", "approved")
#client.set_model_version_tag(MLFLOW_MODEL_NAME, new_model_version, "validation_status", "pending")
# Delete model version tag
#client.delete_model_version_tag(MLFLOW_MODEL_NAME, "1", "validation_status")
