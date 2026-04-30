# Reproducibility Roadmap: Docker Environment

> **Note on Availability:** The containerized infrastructure described in this document is currently a **target setup** under development. The configuration files (`Dockerfile`, `docker-compose.yaml`) are scheduled for a future release to ensure full cross-platform reproducibility.

## Planned Architecture

The intended environment will consist of four orchestrated services to ensure isolation of the **Ψ-Risk-DT** components:

1.  **mqtt-broker**: Eclipse Mosquitto (v2.0) for IoT telemetry transport.
2.  **fuseki**: Apache Jena Fuseki (v4.x) serving as the Semantic Knowledge Graph endpoint.
3.  **digital-twin**: The Python-based neurosymbolic reasoning engine.
4.  **iot-simulator**: A utility to generate synthetic normal and attack traffic patterns.

---

## Targeted Workflow

Once the infrastructure files are integrated, the reproducibility workflow will follow these standard steps:

### 1. Build & Configuration
The setup will rely on an `experiment_config.yaml` file to manage credentials,parameters and ports, followed by the standard build command:

```bash    
# Planned command:
# docker-compose up -d --build
```

## 2. Launch Baseline
To capture normal traffic metrics for the $\Psi$ baseline:

```bash
# Planned command:
# docker exec -it iot-simulator python iot_simulator.py --mode normal
```

## 3. Launch Attack
To simulate a DDoS (NTP Amplification) attack:

```bash
# Planned command:
# docker exec -it iot-simulator python iot_simulator.py --mode attack
```

## 4. Aggregate Results
After execution, the aggregated JSON reports will be generated.
