

# ## Integrate the model with the forecasting application
# 
# Now that you have trained and registered a power forecasting model with the MLflow Model Registry, the next step is to integrate it with an application. This application fetches a weather forecast for the wind farm over the next five days and uses the model to produce power forecasts. For example purposes, the application consists of a simple forecast_power() function (defined below) that is executed within this notebook. In practice, you may want to execute this function as a recurring batch inference job using the Databricks Jobs service.
# 
# The following section demonstrates how to load model versions from the MLflow Model Registry for use in applications. The Forecast power output with the production model section uses the Production model to forecast power output for the next five days.
# 
# ### Load versions of the registered model
# 
# The MLflow Models component defines functions for loading models from several machine learning frameworks. For example, mlflow.tensorflow.load_model() is used to load Tensorflow Keras models that were saved in MLflow format, and mlflow.sklearn.load_model() is used to load scikit-learn models that were saved in MLflow format.
# 
# These functions can load models from the MLflow Model Registry.
# 


# ### Forecast power output with the production model
# 
# In this section, the production model is used to evaluate weather forecast data for the wind farm. The forecast_power() application loads the latest version of the forecasting model from the specified stage and uses it to forecast power production over the next five days.
# 


# #### MLflow part
# 
# **! Configure IMPORTANT CONSTANTS !:**

#set the environmental vars to allow 'mlflow_user' to track experiments using MLFlow
import os
import getpass

# IMPORTANT CONSTANTS TO DEFINE
# MLFLOW CREDENTIALS (Nginx). PUT REAL ONES!
# for direct API calls via HTTP we need to inject credentials
MLFLOW_TRACKING_USERNAME = 'lisana.berberi@kit.edu'
MLFLOW_TRACKING_PASSWORD =  getpass.getpass()  # inject password by typing manually
# for MLFLow-way we have to set the following environment variables
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

# Remote MLFlow server
MLFLOW_REMOTE_SERVER="https://mlflow.dev.ai4eosc.eu" 
# Name of the experiment (e.g. name of the  code repository)
MLFLOW_EXPERIMENT_NAME="wind_power_forecast_L"
# Name of the model to train. HAS TO BE UNIQUE, Please, DEFINE ONE!
MLFLOW_MODEL_NAME="Wind_Forecast_L"


import matplotlib.dates as mdates
from matplotlib import pyplot as plt

def plot(model_name, model_stage, model_version, power_predictions, past_power_output):

  index = power_predictions.index

  fig = plt.figure(figsize=(11, 7))

  ax = fig.add_subplot(111)

  ax.set_xlabel("Date", size=20, labelpad=20)

  ax.set_ylabel("Power\noutput\n(MW)", size=20, labelpad=60, rotation=0)

  ax.tick_params(axis='both', which='major', labelsize=17)

  ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

  ax.plot(index[:len(past_power_output)], past_power_output, label="True", color="red", alpha=0.5, linewidth=4)

  ax.plot(index, power_predictions.squeeze(), "--", label="Predicted by '%s'\nin stage '%s' (Version %d)" % (model_name, model_stage, model_version), color="blue", linewidth=3)

  ax.set_ylim(ymin=0, ymax=max(3500, int(max(power_predictions.values) * 1.3)))

  ax.legend(fontsize=14)

  plt.title("Wind farm power output and projections", size=24, pad=20)

  plt.tight_layout()

  display(plt.show())
  


from mlflow.tracking.client import MlflowClient

def forecast_power(model_name, model_stage):
  client = MlflowClient()
  model_version = client.get_latest_versions(model_name, stages=[model_stage])[0].version
  model_uri = F"models:/{model_name}/{model_stage}"
  model = mlflow.pyfunc.load_model(model_uri)
  weather_data, past_power_output = get_weather_and_forecast()
  power_predictions = pd.DataFrame(model.predict(weather_data))
  power_predictions.index = pd.to_datetime(weather_data.index)
  print(power_predictions)
  plot(model_name, model_stage, int(model_version), power_predictions, past_power_output)

forecast_power(MLFLOW_MODEL_NAME, "Production")

# ## Create and deploy a new model version
# 
# The MLflow Model Registry enables you to create multiple model versions corresponding to a single registered model. By performing stage transitions, you can seamlessly integrate new model versions into your staging or production environments. Model versions can be trained in different machine learning frameworks (such as scikit-learn and tensorflow); MLflow's python_function provides a consistent inference API across machine learning frameworks, ensuring that the same application code continues to work when a new model version is introduced.
# 
# The following sections create a new version of the power forecasting model using scikit-learn, perform model testing in Staging, and update the production application by transitioning the new model version to Production.
# 
#classic ML application ; The following cell trains a random forest model using scikit-learn and 
#registers it with the MLflow Model Registry via the mlflow.sklearn.log_model() function.
##(debug) MLFLOW_REMOTE_SERVER
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

 #Set the MLflow server and backend and artifact stores
mlflow.set_tracking_uri(MLFLOW_REMOTE_SERVER)

#set an experiment name for all different runs
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run(run_name='RandomForestRegressor'):

  n_estimators = 100
  mlflow.log_param("n_estimators", n_estimators)

  run_id = mlflow.active_run().info.run_id
  rand_forest = RandomForestRegressor(n_estimators=n_estimators)
  rand_forest.fit(X_train, y_train)

  val_x, val_y = get_validation_data()
  mse = mean_squared_error(rand_forest.predict(val_x), val_y)
  print("Validation MSE: %d" % mse)
  mlflow.log_metric("mse", mse)
  mlflow.sklearn.autolog()

# The default path where the MLflow autologging function stores the model
model_uri = F"runs:/{run_id}/model"
client = MlflowClient()
# Add newly trained model as new version for our MLFLOW_MODEL_NAME
client.create_model_version(source=model_uri, run_id=run_id, name=MLFLOW_MODEL_NAME)


# ![sklearn_model.jpeg](attachment:sklearn_model.jpeg)

# #### Fetch the new model using model search registry
# 

from mlflow.tracking.client import MlflowClient
client = MlflowClient()
model_version_infos = client.search_model_versions(F"name = '{MLFLOW_MODEL_NAME}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

wait_until_ready(MLFLOW_MODEL_NAME, new_model_version)

client.transition_model_version_stage(
  name=MLFLOW_MODEL_NAME,
  version=new_model_version,
  stage="Staging",
)

forecast_power(MLFLOW_MODEL_NAME, "Staging")

client.transition_model_version_stage(

  name=MLFLOW_MODEL_NAME,
  version=new_model_version,
  stage="Production",

)

# ### Archive and delete models
# 
# When a model version is no longer being used, you can archive it or delete it. You can also delete an entire registered model; this removes all of its associated model versions.
# 


client = MlflowClient()
client.transition_model_version_stage(
  name=MLFLOW_MODEL_NAME,
  version=1,
  stage="Archived",
)


client.delete_model_version(
 name=MLFLOW_MODEL_NAME,
 version=1,
)


# Delete the power forecasting model:
# 

# archive first
client.transition_model_version_stage(
  name=MLFLOW_MODEL_NAME,
  version=2,
  stage="Archived"
)

# then delete
client.delete_registered_model(name=MLFLOW_MODEL_NAME)