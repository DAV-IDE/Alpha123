# Reproducibility Guide: Docker Environment

This document provides instructions to build, configure, and execute the **Ψ-Risk-DT** artifact using Docker Containers. This setup ensures isolation and reproducibility of the experimental results presented in the paper.

## Architecture

The environment consists of four orchestrated services (defined in `docker-compose.yaml`):

1.  **mqtt-broker**: Eclipse Mosquitto (v2.0) handling IoT telemetry transport.
2.  **fuseki**: Apache Jena Fuseki (v4.x) serving as the Semantic Knowledge Graph endpoint.
3.  **digital-twin**: The Python-based neurosymbolic reasoning engine.
4.  **iot-simulator**: A utility to generate synthetic normal and attack traffic patterns.

## Prerequisites

* **Docker Engine** (v20.10 or higher)
* **Docker Compose** (v2.0 or higher)

## Configuration

Environment variables are defined in the `.env` file. Ensure the following defaults are set for local testing before building:

```ini
# Fuseki Credentials
FUSEKI_PASSWORD=admin123
FUSEKI_USER=admin
```

## Build & Startup

To build the images from the source code and start the orchestration in detached mode:

```bash    
docker-compose up -d --build
```
Check the status of the containers to ensure everything is running:

```bash    
docker-compose ps
```
> **Note:** Please wait approximately **10-15 seconds** for the Fuseki server to fully initialize the dataset (`/ds`) before running simulations.

## Running Simulations

You can run simulations in two ways: manually (for specific tests) or automatically (for reproducibility experiments).

### 1. Manual Execution (CLI)

Use the simulator to generate traffic with custom parameters. You can customize `mode`, `interval`, `devices`, and `duration`.

**Option A: Normal Traffic**
Run a standard simulation with a 2-second interval between packets:

```bash    
docker exec -it iot-simulator python iot_simulator.py --mode normal --interval 2.0
```
**Option B: Attack Traffic**
Simulate a network attack scenario:

```bash    
docker exec -it iot-simulator python iot_simulator.py --mode attack --interval 2.0
```
> **Tip:** To terminate a manual simulation early, use `Ctrl+C`.

### 2. Automatic Experiment (Replay)

To run a full simulation based on the settings defined in `experiment_config.yaml`:

```bash    
docker exec -it iot-simulator python replay_simulation.py
```
## Verifying Results & Logs

### Check Generated Files
After the simulation, check the local `results/` folder for artifacts:
* `results/entropy_analysis_ws100.csv`: Timestamped entropy values.
* `results/performance_report.json`: Updated metrics after training/inference.

### Check Container Logs
If you need to debug the Digital Twin logic or see real-time processing:

```bash    
docker-compose logs -f digital-twin
```
You can also check other containers (e.g., `iot-simulator` or `fuseki`).

### Access Knowledge Graph
Navigate to the Fuseki UI to inspect the injected RDF triples.
* **URL:** `http://localhost:3030`
* **Credentials:** User: `admin`, Pass: `admin123`
* **Dataset:** `ds`

## Teardown

To stop all services and preserve data volumes:

```bash    
docker-compose down
```

To remove volumes (resetting the database and logs for a fresh experiment):

```bash
docker-compose down -v
```