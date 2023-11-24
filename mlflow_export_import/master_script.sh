#!/bin/bash

# Set default export directory
DEFAULT_EXPORT_DIR="/tmp/export"

# Function to export experiments
export_import_experiments() {
    ./experiments.sh
}

# Function to export registered models
export_import_models() {
    ./registeredModels.sh
}

# Function to export runs
export_import_runs() {
    ./runs.sh
}

# Function to export registered models versions
export_import_models_versions() {
    ./registeredModelsVersion.sh
}

# Main menu
while true; do
    echo "Select an option:"
    echo "1. Export-Import Experiments"
    echo "2. Export-Import Registered Models"
    echo "3. Export-Import Runs"
    echo "4. Export-Import Registered Models Version"
    echo "5. Exit"

    read -p "Enter your choice (1-5): " choice

    case $choice in
        1) export_import_experiments;;
        2) export_import_models;;
        3) export_import_runs;;
        4) export_import_models_versions;;
        5) exit;;
        *) echo "Invalid choice. Please enter a number between 1 and 9.";;
    esac
done
