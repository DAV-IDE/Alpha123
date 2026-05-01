# Reproducibility Roadmap: Docker Environment

> **Note on Availability:** The containerized infrastructure described in this document is currently a **target setup** under development. The configuration files (`Dockerfile`, `docker-compose.yaml`) are scheduled for a future release to ensure full cross-platform reproducibility.

---

## Targeted Workflow

Once the infrastructure files are integrated, the reproducibility workflow will follow these standard steps:

### 1. Build & Configuration
The setup will rely on an `experiment_config.yaml` file to manage credentials, parameters and ports, followed by the standard build command:

```bash    
# Planned command:
# docker-compose up -d --build
```

### 2. Launch Baseline
To capture normal traffic metrics for the $\Psi$ baseline:

```bash
# Planned command:
# docker exec -it iot-simulator python iot_simulator.py --mode normal
```

### 3. Launch Attack
To simulate a DDoS (NTP Amplification) attack:

```bash
# Planned command:
# docker exec -it iot-simulator python iot_simulator.py --mode attack
```

### 4. Aggregate Results
After execution, the aggregated JSON reports will be generated.
