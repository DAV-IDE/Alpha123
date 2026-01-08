# Ψ-Risk-DT: Neurosymbolic Digital Twin Core

## Overview
This component implements the core **Neurosymbolic Digital Twin** logic described in the reference paper. It serves as the reasoning engine of the architecture, orchestrating the O1-O6 pipeline by integrating an Adaptive Random Neural Network (ARNN) with Semantic Knowledge Graphs for real-time risk assessment in IoT networks.

## Pipeline Implementation
The `digital_twin.py` script executes the following stages sequentially upon receiving MQTT telemetry:

* **O1 - Data Ingestion & Serialization:** Consumes raw MQTT telemetry and converts it into RDF triples (Ontology population).
* **O2 - Entropy-based Gating:** Calculates Shannon Entropy on sliding packet windows. Significant deviations ($\Delta H > \tau_s$) trigger the online learning mechanism.
* **O3 - Vectorization:** Transforms categorical (protocol, ports) and numerical features into tensors for the neural module.
* **O4 - ARNN Inference:** A Recurrent ARNN processes the state to predict a risk score ($0 \in [0,1]$).
* **O5 - Semantic Injection:** High-risk events and neural activations are injected back into the Knowledge Graph (Fuseki) via SPARQL Updates.
* **O6 - Mitigation & Feedback:** Applies defensive policies (simulated latency) and exposes metrics via REST APIs for the UI.

## Dependencies
The module requires Python 3.9+ and the following key libraries (see `requirements.txt`):
* **PyTorch:** For ARNN model construction and gradient descent.
* **RDFLib:** For RDF graph manipulation and N-Triples serialization.
* **Flask:** Exposes health status, real-time metrics, and plotting endpoints.
* **Paho-MQTT:** Handles asynchronous communication with the IoT layer.

## Input / Output Interfaces

### Inputs
* **Protocol:** MQTT (v5)
* **Topic:** `iot/sensor_data/#`
* **Payload Format:** JSON containing `src`, `dst`, `proto`, `size`, `value`, `ts`.

### Outputs
1.  **Semantic Updates:** SPARQL `INSERT DATA` commands sent to the Apache Jena Fuseki endpoint.
2.  **Audit Logs (Volume Persistence):**
    * `results/entropy_analysis_ws{size}.csv`: Entropy trends and threshold violations.
    * `results/detection_times.csv`: Inference latency tracking.
    * `results/mitigation_times.csv`: Log of applied defensive actions.
    * `results/performance_report.json`: Real-time model metrics (AUC, F1, Accuracy).
3.  **API Endpoints (Port 8080):**
    * `GET /health`: Service status and training state.
    * `GET /metrics`: Returns current model performance in JSON.
    * `GET /plots`: Returns paths to generated visualization images.