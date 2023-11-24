# MLFlow Export-Import Options



## Getting started

This is a basic tutorial in exporting-importing mlflow objects either using BULK or SINGLE tool

## Prerequisites

### Main installation
you have:
* python3 > 3.7 but lower than 3.11
* pip is installed

### (Optional) Install favorite virtual environment
For example:
* `sudo apt install python3-virtualenv`
* `python3 -m venv path/to/mlflow-venv`
* `source mlflow-venv/bin/activate`
* (to stop virtualenv) `deactivate`

### Install dependencies from the requirements.txt
* run `pip install -r requirements.txt`

### SINGLE tool 
You can use this approach when you want to export-import between MLflow OS deployments only single mlflow objects and their attributes
* Run command: ` python master_script.py `

### BULK tool 
You can use the following commands in case you want to export-import between MLflow OS deployments ALL mlflow objects
* Run command for exporting ALL: ` export-all --output-dir /tmp/mlflow_objects `
* Run command for importing ALL: ` import-all --input-dir /tmp/mlflow_objects `
* Run command for exporting models: ` export-models --models all --output-dir /tmp/mlflow_objects `
* Run command for importing models: ` import-models --input-dir /tmp/mlflow_objects `

## Acknowledgment
This work is co-funded by AI4EOSC project that has received funding from the European Union's Horizon Europe 2022 research and innovation programme under agreement No 101058593

## License
For open source projects, say how it is licensed.
