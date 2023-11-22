#!/bin/bash

# Set default export directory
DEFAULT_EXPORT_DIR="/tmp/export"


# Prompt user for input to enter info on the original mlflow instance (from)
read -p "Enter MLflow Tracking Server URI: [FROM where you want to get the exps] (e.g., https://mlflow.dev.ai4eosc.eu): " MLFLOW_TRACKING_URI
read -p "Enter MLflow Tracking Username: " MLFLOW_TRACKING_USERNAME
read -s -p "Enter MLflow Tracking Password: " MLFLOW_TRACKING_PASSWORD
echo  # Add a newline after the password input

read -p "Enter the experiment name to export: " EXPERIMENT_NAME

# Prompt user for export directory or use the default
read -p "Enter the local temporary imp/exp directory (default: $DEFAULT_EXPORT_DIR): " EXPORT_OUTPUT_DIR_INPUT
EXPORT_OUTPUT_DIR="${EXPORT_OUTPUT_DIR_INPUT:-$DEFAULT_EXPORT_DIR}"


# Set environment variables
export MLFLOW_TRACKING_URI
export MLFLOW_TRACKING_USERNAME
export MLFLOW_TRACKING_PASSWORD

# Export experiment
export-experiment --experiment $EXPERIMENT_NAME --output-dir $EXPORT_OUTPUT_DIR

# Prompt user for input again

# Prompt user for input to enter info on the destination mlflow instance (to)
read -p "Enter MLflow Tracking Server URI: [TO where you want to save the exps] (e.g., http://localhost:5000): " MLFLOW_TRACKING_URI
read -p "Enter MLflow Tracking Username: " MLFLOW_TRACKING_USERNAME
read -s -p "Enter MLflow Tracking Password: " MLFLOW_TRACKING_PASSWORD
echo  # Add a newline after the password input

# Set the new environment variables
export MLFLOW_TRACKING_URI
export MLFLOW_TRACKING_USERNAME
export MLFLOW_TRACKING_PASSWORD


read -p "Enter the experiment name to import: " EXPERIMENT_NAME
# Prompt user for import directory or use the default
read -p "Enter the local temporary imp/exp directory (default: $DEFAULT_EXPORT_DIR): " IMPORT_INPUT_DIR_INPUT
IMPORT_INPUT_DIR="${IMPORT_INPUT_DIR_INPUT:-$DEFAULT_EXPORT_DIR}"

# Import experiment
import-experiment --experiment-name $EXPERIMENT_NAME --input-dir $IMPORT_INPUT_DIR

# Clean up: unset environment variables
unset MLFLOW_TRACKING_URI
unset MLFLOW_TRACKING_USERNAME
unset MLFLOW_TRACKING_PASSWORD