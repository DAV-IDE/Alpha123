"""
DIGITAL TWIN - Complete Pipeline Implementation O1-O6
"""

import json
import warnings
import numpy as np
import paho.mqtt.client as mqtt
from rdflib import Graph, Namespace, Literal, RDF, URIRef, XSD
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify
import threading
import time
from collections import deque
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import scipy.stats as stats
import yaml
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ======================= CONFIGURATION =======================
try:
    with open('/app/config/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    RANDOM_SEED = config['experiment']['seed']
    ENTROPY_WINDOW_SIZES = config['entropy']['window_sizes']
    PERCENTILE_THRESHOLD = config['entropy']['percentile_threshold']
    BASELINE_DURATION = config['entropy']['baseline_duration']
    print(f"[CONFIG] Configuration loaded with seed {RANDOM_SEED}")
except Exception as e:
    print(f"[CONFIG ERROR] Using default configuration: {e}")
    RANDOM_SEED = 42
    ENTROPY_WINDOW_SIZES = [100, 200, 500]
    PERCENTILE_THRESHOLD = 95
    BASELINE_DURATION = 300

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ======================= GLOBAL VARIABLES =======================
training_in_progress = False
startup_time = time.time()

# ======================= O1 - RDF SERIALIZATION =======================
NTW = Namespace("http://example.org/network#")

def packet_to_rdf(payload, risk="low", score=None):
    g = Graph()
    pkt_uri = URIRef(f"http://example.org/packet/{payload['device']}_{payload['ts']}")

    g.add((pkt_uri, RDF.type, NTW.Packet))
    g.add((pkt_uri, NTW.src, Literal(payload["src"])))
    g.add((pkt_uri, NTW.dst, Literal(payload["dst"])))
    g.add((pkt_uri, NTW.proto, Literal(payload["proto"])))
    g.add((pkt_uri, NTW.port, Literal(payload["port"], datatype=XSD.integer)))
    g.add((pkt_uri, NTW.size, Literal(payload["size"], datatype=XSD.integer)))
    g.add((pkt_uri, NTW.timestamp, Literal(payload["ts"], datatype=XSD.long)))
    g.add((pkt_uri, NTW.risk, Literal(risk)))

    if score is not None:
        g.add((pkt_uri, NTW.hasRiskScore, Literal(score, datatype=XSD.float)))

    return g

def send_to_fuseki(graph):
    data = graph.serialize(format="nt")
    try:
        response = requests.post(
            "http://fuseki:3030/ds/data",
            data=data,
            headers={"Content-Type": "application/n-triples"},
            timeout=3
        )
        if response.status_code not in (200, 201, 204):
            print(f"[ERROR] Fuseki insert failed: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Cannot send to Fuseki: {e}")

# ======================= O2 - ENTROPY & THRESHOLDING =======================
entropy_config = {
    'window_sizes': ENTROPY_WINDOW_SIZES,
    'baseline_established': False,
    'percentile_threshold': PERCENTILE_THRESHOLD,
    'dynamic_threshold': None,
    'start_time': None
}

packet_windows = {size: deque(maxlen=size) for size in ENTROPY_WINDOW_SIZES}
entropy_histories = {size: [] for size in ENTROPY_WINDOW_SIZES}
H_previous_values = {size: 0 for size in ENTROPY_WINDOW_SIZES}
delta_entropy_history = {size: [] for size in ENTROPY_WINDOW_SIZES}

def calculate_entropy(sizes):
    if len(sizes) == 0:
        return 0
    hist, _ = np.histogram(sizes, bins=10, range=(0, 1500))
    prob = hist / (hist.sum() + 1e-10)
    return -np.sum(prob * np.log2(prob + 1e-10))

def entropy_based_detection(payload):
    global entropy_config, packet_windows, entropy_histories, H_previous_values, delta_entropy_history

    alarms = {}
    deltas = {}
    packet_time = payload['ts'] / 1000.0

    for window_size in entropy_config['window_sizes']:
        packet_windows[window_size].append(payload['size'])

        if len(packet_windows[window_size]) == window_size:
            H_current = calculate_entropy(packet_windows[window_size])

            delta_H = abs(H_current - H_previous_values[window_size])

            H_previous_values[window_size] = H_current
            entropy_histories[window_size].append(H_current)
            delta_entropy_history[window_size].append(delta_H)

            time_since_start = packet_time - entropy_config['start_time']

            if time_since_start < BASELINE_DURATION:
                default_threshold = 1.0
            else:
                default_threshold = 0.8

            if (time_since_start >= BASELINE_DURATION and
                not entropy_config['baseline_established'] and
                entropy_config['dynamic_threshold'] is None):

                if len(delta_entropy_history[window_size]) > 50:
                    threshold = np.percentile(delta_entropy_history[window_size],
                                             entropy_config['percentile_threshold'])
                    entropy_config['dynamic_threshold'] = threshold
                    entropy_config['baseline_established'] = True
                    print(f"[ENTROPY] Baseline established for window {window_size}")
                    print(f"[ENTROPY] Threshold (P{entropy_config['percentile_threshold']}): {threshold:.3f}")

            threshold = (entropy_config['dynamic_threshold']
                        if entropy_config['dynamic_threshold'] is not None
                        else default_threshold)

            alarm = delta_H > threshold
            alarms[window_size] = alarm
            deltas[window_size] = delta_H

            # Writing entropy analysis to CSV (Audit Trail)
            with open(f'/app/results/entropy_analysis_ws{window_size}.csv', 'a') as f:
                f.write(f"{packet_time:.3f},{window_size},{H_current:.6f},{delta_H:.6f},"
                       f"{threshold:.6f},{alarm}\n")

            if alarm:
                print(f"[ENTROPY ALARM] Window {window_size}: ΔH={delta_H:.3f} > {threshold:.3f}")

    any_alarm = any(alarms.values())
    return any_alarm, deltas

# ======================= O3 - VECTORIZATION =======================
PROTOCOLS = ['UDP', 'TCP', 'ICMP']
COMMON_PORTS = [80, 443, 123, 53, 8080, 22, 21]

def vectorize(payload):
    proto_onehot = [1 if payload["proto"] == p else 0 for p in PROTOCOLS]
    port_onehot = [1 if payload["port"] == p else 0 for p in COMMON_PORTS]
    normalized_size = payload["size"] / 1500.0
    normalized_value = payload["value"] / 100.0

    return np.array(proto_onehot + port_onehot + [normalized_size, normalized_value],
                   dtype=np.float32)

# ======================= O4 - ARNN CORE =======================
# ARNN (Adaptive Random Neural Network) implementation - Note: This is an A-RNN (Adaptive-RNN) architecture,
# not a classic Random Neural Network.
class ARNN(nn.Module):
    def __init__(self, d_in, d_hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

arnn = ARNN(d_in=12)
criterion = nn.BCELoss()
optimizer = optim.Adam(arnn.parameters(), lr=0.001, weight_decay=1e-5)

X_train, y_train = [], []
fitted = False
training_mode = "normal"
training_samples = 0

performance_metrics = {
    'training_start_time': None,
    'inference_times': [],
    'entropy_alarms': 0,
    'detection_times': [],
    'mitigation_times': []
}

def calculate_confidence_interval(data, confidence=0.95):
    if len(data) < 2:
        return np.mean(data), 0, 0

    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

def calculate_performance_metrics(y_true, y_pred, y_scores):
    try:
        auc = roc_auc_score(y_true, y_scores)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        n_bootstraps = 1000
        auc_scores = []
        rng = np.random.RandomState(RANDOM_SEED)

        for i in range(n_bootstraps):
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2:
                continue
            auc_bootstrap = roc_auc_score(y_true[indices], y_scores[indices])
            auc_scores.append(auc_bootstrap)

        auc_mean, auc_low, auc_high = calculate_confidence_interval(auc_scores)

        detection_stats = {}
        if performance_metrics['detection_times']:
            det_mean, det_low, det_high = calculate_confidence_interval(performance_metrics['detection_times'])
            detection_stats = {
                'mean': round(det_mean, 2),
                'ci_95_low': round(det_low, 2),
                'ci_95_high': round(det_high, 2)
            }

        mitigation_stats = {}
        if performance_metrics['mitigation_times']:
            mit_mean, mit_low, mit_high = calculate_confidence_interval(performance_metrics['mitigation_times'])
            mitigation_stats = {
                'mean': round(mit_mean, 2),
                'ci_95_low': round(mit_low, 2),
                'ci_95_high': round(mit_high, 2)
            }

        return {
            'auc': {
                'value': round(auc, 4),
                'ci_95_low': round(auc_low, 4),
                'ci_95_high': round(auc_high, 4)
            },
            'f1_score': round(f1, 4),
            'false_positive_rate': round(fpr, 4),
            'accuracy': round(accuracy, 4),
            'confusion_matrix': {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            },
            'detection_time_ms': detection_stats,
            'mitigation_time_ms': mitigation_stats
        }
    except Exception as e:
        print(f"[METRICS ERROR] {e}")
        return None

def save_performance_report(y_true, y_pred, y_scores):
    metrics = calculate_performance_metrics(y_true, y_pred, y_scores)

    inf_time_stats = {}
    if performance_metrics['inference_times']:
        inf_time_mean, inf_time_low, inf_time_high = calculate_confidence_interval(
            performance_metrics['inference_times'])
        inf_time_stats = {
            'mean': round(inf_time_mean, 2),
            'ci_95_low': round(inf_time_low, 2),
            'ci_95_high': round(inf_time_high, 2)
        }

    report = {
        'timestamp': time.time(),
        'random_seed': RANDOM_SEED,
        'training_duration': time.time() - performance_metrics['training_start_time'] if performance_metrics[
            'training_start_time'] else 0,
        'inference_time_ms': inf_time_stats,
        'entropy_alarms': performance_metrics['entropy_alarms'],
        'dataset_stats': {
            'total_samples': len(X_train),
            'normal_samples': sum(1 for y in y_train if y == 0),
            'attack_samples': sum(1 for y in y_train if y == 1)
        },
        'model_metrics': metrics
    }

    with open('/app/results/performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)

def generate_plots():
    try:
        for window_size in ENTROPY_WINDOW_SIZES:
            try:
                df = pd.read_csv(f'/app/results/entropy_analysis_ws{window_size}.csv')

                plt.figure(figsize=(12, 6))
                plt.plot(df['timestamp'], df['delta_entropy'],
                        label='|ΔH(t)|', linewidth=1.5, color='blue')

                threshold_value = df['threshold'].iloc[-1] if not df.empty else 0
                plt.axhline(y=threshold_value, color='red',
                           linestyle='--', linewidth=2,
                           label=f'Threshold (P{PERCENTILE_THRESHOLD}) = {threshold_value:.3f}')

                alarms = df[df['alarm'] == True]
                if not alarms.empty:
                    plt.scatter(alarms['timestamp'], alarms['delta_entropy'],
                               color='red', s=50, zorder=5, label='Alarms')

                plt.xlabel('Relative Time (s)')
                plt.ylabel('|Δ Entropy|')
                plt.title(f'Entropy Trend - Window {window_size} packets')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'/app/results/entropy_timeline_ws{window_size}.png',
                           dpi=300, bbox_inches='tight')
                plt.close()

                print(f"[PLOT] Generated entropy timeline for window {window_size}")

            except Exception as e:
                print(f"[PLOT ERROR] Entropy timeline for window {window_size}: {e}")

        if os.path.exists('/app/results/performance_report.json'):
            with open('/app/results/performance_report.json', 'r') as f:
                report = json.load(f)
                if 'model_metrics' in report and report['model_metrics']:
                    plt.figure(figsize=(8, 8))
                    plt.text(0.5, 0.5, f"AUC: {report['model_metrics']['auc']['value']}",
                            ha='center', va='center', fontsize=12)
                    plt.title('ROC Curve')
                    plt.savefig('/app/results/roc_curve.png')
                    plt.close()

        print("[PLOTS] All graphs generated successfully")
    except Exception as e:
        print(f"[PLOTS ERROR] {e}")

def train_model_async(X_train, y_train, normal_samples, attack_samples):
    def training_thread():
        global fitted, arnn, criterion, optimizer, performance_metrics, training_in_progress

        try:
            performance_metrics['training_start_time'] = time.time()
            print(f"[ARNN] Starting training with {len(X_train)} samples")

            total = normal_samples + attack_samples
            weight_for_0 = total / (2 * normal_samples) if normal_samples > 0 else 1
            weight_for_1 = total / (2 * attack_samples) if attack_samples > 0 else 1

            print(f"[ARNN] Class weights - Normal: {weight_for_0:.2f}, Attack: {weight_for_1:.2f}")

            X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
            y_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)

            arnn.train()
            for epoch in range(200):
                optimizer.zero_grad()
                outputs = arnn(X_tensor)

                if weight_for_1 > 1.0:
                    weights_tensor = torch.ones_like(y_tensor) * weight_for_0
                    weights_tensor[y_tensor == 1] = weight_for_1
                    loss = (weights_tensor * criterion(outputs, y_tensor)).mean()
                else:
                    loss = criterion(outputs, y_tensor)

                loss.backward()
                optimizer.step()

                if epoch % 40 == 0:
                    print(f"[ARNN] Epoch {epoch}, Loss: {loss.item():.6f}")

            arnn.eval()
            with torch.no_grad():
                train_outputs = arnn(X_tensor)
                train_predictions = (train_outputs > 0.5).float()
                accuracy = (train_predictions == y_tensor).float().mean()
                print(f"[ARNN] Training completed. Final loss: {loss.item():.6f}, Accuracy: {accuracy.item():.4f}")

            fitted = True
            training_in_progress = False
            print(f"[ARNN] MODEL TRAINING COMPLETED! Switching to inference mode.")

            y_true = y_tensor.cpu().numpy()
            y_pred = (train_outputs > 0.5).float().cpu().numpy()
            y_scores = train_outputs.cpu().numpy()

            save_performance_report(y_true, y_pred, y_scores)
            generate_plots()

        except Exception as e:
            print(f"[TRAINING ERROR] {e}")
            training_in_progress = False

    thread = threading.Thread(target=training_thread, daemon=True)
    thread.start()

def run_sparql_query(query, query_name="query", params=None):
    start_time = time.time()

    if params:
        for key, value in params.items():
            query = query.replace(f"{{{{{key}}}}}", str(value))

    try:
        response = requests.post(
            "http://fuseki:3030/ds/sparql",
            data=query,
            headers={"Content-Type": "application/sparql-query"},
            timeout=10
        )
        latency = (time.time() - start_time) * 1000

        # Writing query performance to JSONL (Audit Trail)
        with open('/app/results/query_performance.jsonl', 'a') as f:
            f.write(json.dumps({
                'timestamp': time.time(),
                'query': query_name,
                'latency_ms': round(latency, 2),
                'status': 'success',
                'params': params
            }) + '\n')

        return response, latency

    except Exception as e:
        latency = (time.time() - start_time) * 1000
        # Writing query performance error to JSONL (Audit Trail)
        with open('/app/results/query_performance.jsonl', 'a') as f:
            f.write(json.dumps({
                'timestamp': time.time(),
                'query': query_name,
                'latency_ms': round(latency, 2),
                'status': 'error',
                'error': str(e),
                'params': params
            }) + '\n')
        return None, latency

def execute_diagnostic_queries():
    # Diagnostic queries are used for semantic reasoning verification and audit trail
    try:
        os.makedirs('/app/results/sparql_queries', exist_ok=True)

        with open('/app/sparql/high-risk-window.rq', 'r') as f:
            query1 = f.read()

        current_time = int(time.time() * 1000)
        one_hour_ago = current_time - (60 * 60 * 1000)

        response1, latency1 = run_sparql_query(
            query1,
            "high_risk_hosts_window",
            {'start_time': one_hour_ago, 'end_time': current_time}
        )

        if response1 and response1.status_code == 200:
            with open('/app/results/sparql_queries/high_risk_hosts.json', 'w') as f:
                json.dump(response1.json(), f, indent=2)

        print(f"[SPARQL] High risk hosts query latency: {latency1:.2f}ms")

        with open('/app/sparql/rpl-loops.rq', 'r') as f:
            query2 = f.read()

        response2, latency2 = run_sparql_query(query2, "rpl_loops")

        if response2 and response2.status_code == 200:
            with open('/app/results/sparql_queries/rpl_loops.json', 'w') as f:
                json.dump(response2.json(), f, indent=2)

        print(f"[SPARQL] RPL loops query latency: {latency2:.2f}ms")

        return True

    except Exception as e:
        print(f"[SPARQL ERROR] Diagnostic queries failed: {e}")
        return False

# Flask server configuration (for health check and metrics)
app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "digital-twin",
        "model_ready": fitted,
        "samples_collected": len(X_train),
        "entropy_windows": {ws: len(packet_windows[ws]) for ws in entropy_config['window_sizes']},
        "inference_count": len(performance_metrics['inference_times']),
        "random_seed": RANDOM_SEED
    })

@app.route('/metrics')
def metrics_endpoint():
    try:
        with open('/app/results/performance_report.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except:
        return jsonify({"error": "Metrics not available yet"})

@app.route('/plots')
def plots_endpoint():
    try:
        plots = {}
        for window_size in ENTROPY_WINDOW_SIZES:
            plot_path = f'/app/results/entropy_timeline_ws{window_size}.png'
            if os.path.exists(plot_path):
                plots[f'entropy_ws{window_size}'] = f'http://localhost:8080/static/entropy_timeline_ws{window_size}.png'

        if os.path.exists('/app/results/roc_curve.png'):
            plots['roc_curve'] = 'http://localhost:8080/static/roc_curve.png'

        return jsonify(plots)
    except Exception as e:
        return jsonify({"error": str(e)})

def run_health_server():
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

def apply_mitigation_policy(src_ip, risk_score):
    # Mitigation policy simulation
    mitigation_start = time.time()
    time.sleep(0.01) # Simulate network latency or policy enforcement time
    mitigation_time = (time.time() - mitigation_start) * 1000
    performance_metrics['mitigation_times'].append(mitigation_time)

    # Writing mitigation action to CSV (Audit Trail)
    with open('/app/results/mitigation_times.csv', 'a') as f:
        f.write(f"{time.time()},{mitigation_time},{src_ip},{risk_score}\n")

    print(f"[MITIGATION] Applied policy for {src_ip} (score: {risk_score:.3f}) in {mitigation_time:.2f}ms")

    return mitigation_time

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe("iot/sensor_data/#")

def on_message(client, userdata, message):
    global fitted, training_mode, training_samples, arnn, criterion, optimizer
    global performance_metrics, training_in_progress, entropy_config

    try:
        payload = json.loads(message.payload.decode())

        if entropy_config['start_time'] is None:
            entropy_config['start_time'] = payload['ts'] / 1000.0
            print(f"[ENTROPY] Start time synchronized with first packet: {entropy_config['start_time']}")

        entropy_alarm, delta_H = entropy_based_detection(payload)
        if entropy_alarm:
            performance_metrics['entropy_alarms'] += 1
            print(f"[ENTROPY ALARM] ΔH = {delta_H}")

        x = vectorize(payload)

        if not fitted:
            # Simple heuristic for labeling during baseline (size > 300 is treated as ATTACK)
            if payload["size"] > 300:
                label = 1
                current_mode = "attack"
            else:
                label = 0
                current_mode = "normal"

            if current_mode != training_mode:
                training_mode = current_mode
                print(f"[ARNN] Switching to {training_mode} mode training")

            X_train.append(x)
            y_train.append(label)
            training_samples += 1

            normal_samples = sum(1 for y in y_train if y == 0)
            attack_samples = sum(1 for y in y_train if y == 1)

            print(f"[ARNN] Collected {training_samples} samples")
            print(f"[DEBUG] Normal: {normal_samples}, Attack: {attack_samples}")

            # Condition to start asynchronous training
            if normal_samples >= 100 and attack_samples >= 100 and not training_in_progress:
                training_in_progress = True
                print(f"[TRAINING] Starting training")
                train_model_async(X_train, y_train, normal_samples, attack_samples)
                return

            g = packet_to_rdf(payload, risk="low")
            send_to_fuseki(g)

        else:
            # O4 - INFERENCE PHASE
            inference_start = time.time()

            x_tensor = torch.tensor(x).unsqueeze(0)
            with torch.no_grad():
                score = float(arnn(x_tensor).item())

            inference_time = (time.time() - inference_start) * 1000
            performance_metrics['inference_times'].append(inference_time)

            print(f"[INFERENCE] Prediction score: {score:.6f}, Time: {inference_time:.2f}ms")

            risk_label = "high" if score > 0.5 else "low"

            # Writing detection time to CSV
            with open('/app/results/detection_times.csv', 'a') as f:
                f.write(f"{payload['ts']},{inference_time}\n")

            # O5 - SEMANTIC GRAPH INJECTION
            g = packet_to_rdf(payload, risk=risk_label, score=score)
            send_to_fuseki(g)

            # O6 - DYNAMIC UPDATE LOOP (Diagnostic Queries)
            if random.random() < 0.1:
                execute_diagnostic_queries()

            if risk_label == "high":
                print(f"[ALERT] Packet {payload['device']} HIGH RISK: {score:.3f}")
                apply_mitigation_policy(payload['src'], score)

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    os.makedirs('/app/results', exist_ok=True)
    os.makedirs('/app/results/sparql_queries', exist_ok=True)

    # Initialize CSV files with headers
    for window_size in entropy_config['window_sizes']:
        with open(f'/app/results/entropy_analysis_ws{window_size}.csv', 'w') as f:
            f.write("timestamp,window_size,entropy,delta_entropy,threshold,alarm\n")

    with open('/app/results/query_performance.jsonl', 'w') as f:
        f.write("")

    with open('/app/results/detection_times.csv', 'w') as f:
        f.write("timestamp,detection_time_ms\n")

    with open('/app/results/mitigation_times.csv', 'w') as f:
        f.write("timestamp,mitigation_time_ms,src_ip,risk_score\n")

    print(f"[SYSTEM] Starting Digital Twin with random seed: {RANDOM_SEED}")
    print(f"[SYSTEM] Entropy window sizes: {entropy_config['window_sizes']}")
    print(f"[SYSTEM] Percentile threshold: P{entropy_config['percentile_threshold']}")
    print(f"[SYSTEM] Baseline duration: {BASELINE_DURATION}s")

    # Start Flask Health Server (separate thread)
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    # Start MQTT client
    client = mqtt.Client(client_id="digital-twin", protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("mqtt-broker", 1883)
    client.loop_forever()