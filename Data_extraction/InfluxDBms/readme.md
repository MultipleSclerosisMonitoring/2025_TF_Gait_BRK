# InfluxDBms

## Introduction
This folder contains the tools and configurations needed to interact with an InfluxDB database. It includes scripts for querying and adding data, as well as specific configurations for the connection.

The main goal of this folder is to simplify access to and management of data stored in InfluxDB for projects related to data analysis and processing.

## Setup – Install
1. Ensure you have the following prerequisites installed:
   - **Python 3.8+**
   - The following Python libraries:
     - `pandas`
     - `influxdb-client`
     - `pyyaml`
     - `urllib3`
   - You can install these dependencies by running:
     ```bash
     pip install pandas influxdb-client pyyaml urllib3
     ```

2. Configure your environment:
   - Ensure the file `config_db.yaml` contains the correct credentials and parameters for your InfluxDB instance.
   - For more information on how to configure this file, see the [Comments](#comments) section.

3. Run tests (optional):
   - To test and run the scripts, you can use:
     ```bash
     python <script_name>.py
     ```

## Usage
This folder includes the following files and their purposes:

- **`cInfluxDB.py`**:  
  Main class that manages the connection and queries to InfluxDB. It provides methods to:
  - Query data within a specific time range and pivot the results (`query_data`).
  - Query data with an adjustable aggregation window (`query_with_aggregate_window`).

- **`fecha_verbose.py`**:  
  A script that supports argument parsing and configuration for batch processes, with adjustable verbosity levels.

- **`orden.sh`**:  
  A script file with instructions to install additional dependencies required to interact with InfluxDB.

- **`config_db.yaml`**:  
  Configuration file containing the credentials and parameters required to connect to the InfluxDB database.
