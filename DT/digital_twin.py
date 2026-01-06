"""
DIGITAL TWIN - Complete Pipeline Implementation O1-O6
Integrated with Ψ-Risk-DT framework: Recurrent ARNN, hybrid loss, entropy gating.
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
from requests.auth import HTTPBasicAuth

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ======================= AUTHENTICATION & CONFIG =======================
def fuseki_auth():
    """Retrieves Fuseki credentials from environment variables."""
    user = os.getenv("FUSEKI_USER")
    pwd = os.getenv("FUSEKI_PASSWORD")
    if user and pwd:
        return HTTPBasicAuth(user, pwd)
    return None

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
packet_counter = 0  # For periodic MSU sync

# Neurosymbolic params
tau_s = 0.15  # Entropy threshold (Ψ-Gating)
theta = 0.7   # Activation threshold
alpha, beta, gamma = 0.4, 0.3, 0.3  # Hybrid loss weights
input_size = 12
hidden_size = 64
prev_a = torch.zeros(hidden_size)  # Initial ARNN state (recurrence)
G = Graph()  # Local graph for MSU (accumulation)
W_gt = torch.randn(hidden_size, hidden_size)  # Placeholder teacher weights

# Global Metrics Storage
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

# ======================= O1 - RDF SERIALIZATION =======================
NTW = Namespace("http://example.org/network#")

def packet_to_rdf(payload, risk="low", score=None, mean_a_t=None):
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
    if mean_a_t is not None:
        # Ψ-Injection: record neural activation in the graph
        g.add((pkt_uri, NTW.risk_activation, Literal(mean_a_t, datatype=XSD.float)))

    return g

def send_to_fuseki(graph):
    # Serialize to N-Triples for INSERT DATA
    triples_nt = graph.serialize(format="nt")
    update = f"INSERT DATA {{ {triples_nt} }}"

    try:
        response = requests.post(
            "http://fuseki:3030/ds/update",
            data=update,
            headers={"Content-Type": "application/sparql-update"},
            auth=fuseki_auth(),  # Authentication
            timeout=5
        )
        if response.status_code not in (200, 201, 204):
            print(f"[ERROR] Fuseki insert failed: {response.status_code} - {response.text[:200]}")
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

            # Dynamic threshold logic
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
                    print(f"[ENTROPY] Baseline established for window {window_size}: {threshold:.3f}")

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
    max_delta_H = max(deltas.values()) if deltas else 0  # Per gating
    return any_alarm, max_delta_H

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
class ARNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Recurrent layer (RNN) instead of simple Linear
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # Association layer
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, prev_a):
        _, hn = self.rnn(x.unsqueeze(0))  # hn shape: (1, hidden_size)
        hn = hn.squeeze(0)
        associated = torch.matmul(prev_a, self.W) + self.b
        combined = torch.cat((hn, associated), dim=0)
        out = torch.sigmoid(self.fc(self.dropout(combined)))
        return out, hn  # Return score and new state

arnn = ARNN(input_size=input_size, hidden_size=hidden_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(arnn.parameters(), lr=0.001, weight_decay=1e-5)

def compute_hybrid_loss(score, y_true, W, W_gt, G, alpha=alpha, beta=beta, gamma=gamma):
    # L_cls: BCE
    L_cls = criterion(score, y_true)
    # L_graph: Frobenius norm (Structural consistency)
    L_graph = torch.norm(W - W_gt, p='fro') ** 2
    # L_sem: Semantic alignment
    risk_activations = []
    for s, p, o in G.triples((None, NTW.risk_activation, None)):
        risk_activations.append(float(o))
    embeddings_graph = np.mean(risk_activations) if risk_activations else 0.0
    L_sem = (score.item() - embeddings_graph) ** 2

    return alpha * L_cls + beta * L_graph + gamma * L_sem

# ======================= AUXILIARY & METRICS FUNCTIONS =======================

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

        # Detection stats
        detection_stats = {}
        if performance_metrics['detection_times']:
            det_mean, det_low, det_high = calculate_confidence_interval(performance_metrics['detection_times'])
            detection_stats = {'mean': round(det_mean, 2), 'ci_95_low': round(det_low, 2), 'ci_95_high': round(det_high, 2)}

        mitigation_stats = {}
        if performance_metrics['mitigation_times']:
            mit_mean, mit_low, mit_high = calculate_confidence_interval(performance_metrics['mitigation_times'])
            mitigation_stats = {'mean': round(mit_mean, 2), 'ci_95_low': round(mit_low, 2), 'ci_95_high': round(mit_high, 2)}

        return {
            'auc': {'value': round(auc, 4)},
            'f1_score': round(f1, 4),
            'false_positive_rate': round(fpr, 4),
            'accuracy': round(accuracy, 4),
            'detection_time_ms': detection_stats,
            'mitigation_time_ms': mitigation_stats
        }
    except Exception as e:
        print(f"[METRICS ERROR] {e}")
        return None

def save_performance_report(y_true, y_pred, y_scores):
    metrics = calculate_performance_metrics(y_true, y_pred, y_scores)

    report = {
        'timestamp': time.time(),
        'random_seed': RANDOM_SEED,
        'entropy_alarms': performance_metrics['entropy_alarms'],
        'model_metrics': metrics
    }

    with open('/app/results/performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)

def generate_plots():
    try:
        for window_size in ENTROPY_WINDOW_SIZES:
            try:
                csv_path = f'/app/results/entropy_analysis_ws{window_size}.csv'
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    plt.figure(figsize=(12, 6))
                    plt.plot(df['timestamp'], df['delta_entropy'], label='|ΔH(t)|', color='blue')

                    # Threshold line
                    threshold = df['threshold'].iloc[-1] if not df.empty else 0
                    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold {threshold:.3f}')

                    plt.title(f'Entropy Trend - Window {window_size}')
                    plt.legend()
                    plt.savefig(f'/app/results/entropy_timeline_ws{window_size}.png')
                    plt.close()
            except Exception as e:
                print(f"[PLOT ERROR] {e}")
    except Exception as e:
        print(f"[PLOTS ERROR] {e}")

# ======================= DIAGNOSTIC & QUERY FUNCTIONS =======================

def run_sparql_query(query, query_name="query", params=None):
    start_time = time.time()
    if params:
        for key, value in params.items():
            query = query.replace(f"{{{{{key}}}}}", str(value))

    try:
        # Authentication update
        response = requests.post(
            "http://fuseki:3030/ds/sparql",
            data=query,
            headers={"Content-Type": "application/sparql-query"},
            auth=fuseki_auth(),
            timeout=10
        )
        latency = (time.time() - start_time) * 1000

        # Log query performance
        with open('/app/results/query_performance.jsonl', 'a') as f:
            status = 'success' if response.status_code == 200 else 'error'
            f.write(json.dumps({
                'timestamp': time.time(),
                'query': query_name,
                'latency_ms': latency,
                'status': status
            }) + '\n')

        return response, latency
    except Exception as e:
        print(f"[SPARQL ERROR] {e}")
        return None, 0

def execute_diagnostic_queries():
    try:
        os.makedirs('/app/results/sparql_queries', exist_ok=True)
        # Dummy query example
        query_rpl = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 5"
        resp, lat = run_sparql_query(query_rpl, "diagnostic_check")
        if resp and resp.status_code == 200:
            print(f"[SPARQL DIAGNOSTIC] Check passed in {lat:.2f}ms")
    except Exception as e:
        print(f"[DIAGNOSTIC ERROR] {e}")

def apply_mitigation_policy(src_ip, risk_score):
    mitigation_start = time.time()
    time.sleep(0.01) # Simulate enforcement latency
    mitigation_time = (time.time() - mitigation_start) * 1000
    performance_metrics['mitigation_times'].append(mitigation_time)

    with open('/app/results/mitigation_times.csv', 'a') as f:
        f.write(f"{time.time()},{mitigation_time},{src_ip},{risk_score}\n")

    print(f"[MITIGATION] Policy applied for {src_ip} (Score: {risk_score:.2f})")

# ======================= TRAINING LOOP =======================

def train_model_async(X_train, y_train, normal_samples, attack_samples):
    def training_thread():
        global fitted, arnn, criterion, optimizer, performance_metrics, training_in_progress

        try:
            performance_metrics['training_start_time'] = time.time()
            print(f"[ARNN] Starting training with {len(X_train)} samples")

            # Training sequence logic
            arnn.train()
            local_state = torch.zeros(hidden_size)

            for epoch in range(200):
                epoch_loss = 0
                for i in range(len(X_train)):
                    optimizer.zero_grad()
                    x_tensor = torch.tensor(X_train[i], dtype=torch.float32).unsqueeze(0)
                    y_tensor = torch.tensor([[y_train[i]]], dtype=torch.float32)

                    score, new_state = arnn(x_tensor, local_state)
                    local_state = new_state.detach()

                    # HYBRID LOSS
                    L_total = compute_hybrid_loss(score, y_tensor, arnn.W, W_gt, G)
                    L_total.backward()
                    optimizer.step()
                    epoch_loss += L_total.item()

                if epoch % 20 == 0:
                    avg_loss = epoch_loss / len(X_train)
                    print(f"[ARNN TRAINING] Epoch {epoch}/200 | Hybrid Loss: {avg_loss:.6f}")

            arnn.eval()
            fitted = True
            training_in_progress = False
            print(f"[ARNN] TRAINING COMPLETED. Switching to inference.")

            # Generate report
            y_pred_dummy = [0] * len(y_train) # Placeholder
            save_performance_report(y_train, y_pred_dummy, y_train)
            generate_plots()

        except Exception as e:
            print(f"[TRAINING ERROR] {e}")
            training_in_progress = False

    thread = threading.Thread(target=training_thread, daemon=True)
    thread.start()

# ======================= MQTT & MAIN LOGIC =======================

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe("iot/sensor_data/#")


def on_message(client, userdata, message):
    global fitted, training_mode, training_samples, arnn, optimizer
    global performance_metrics, training_in_progress, entropy_config, prev_a, G, packet_counter

    try:
        payload = json.loads(message.payload.decode())

        if entropy_config['start_time'] is None:
            entropy_config['start_time'] = payload['ts'] / 1000.0

        # 1. Entropy
        entropy_alarm, delta_H = entropy_based_detection(payload)

        # 2. Vectorize
        x = vectorize(payload)

        # 3. Training Logic (Initial phase)
        if not fitted:
            if payload["size"] > 300: # Heuristic Labeling
                label = 1
            else:
                label = 0

            X_train.append(x)
            y_train.append(label)
            training_samples += 1

            normal_s = sum(1 for y in y_train if y == 0)
            attack_s = sum(1 for y in y_train if y == 1)

            if normal_s >= 100 and attack_s >= 100 and not training_in_progress:
                training_in_progress = True
                train_model_async(X_train, y_train, normal_s, attack_s)

            elif training_samples % 20 == 0:
                print(f"[DATASET] Collecting... Total: {training_samples} (Normal: {normal_s}, Attack: {attack_s})")

            # Basic RDF Log
            g = packet_to_rdf(payload, risk="low")
            G += g

        else:
            # 4. INFERENCE (Stateful)
            inference_start = time.time()
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                score, new_a = arnn(x_tensor, prev_a)
            prev_a = new_a

            mean_a_t = torch.mean(new_a).item()
            inference_time = (time.time() - inference_start) * 1000
            performance_metrics['inference_times'].append(inference_time)

            risk_label = "high" if score.item() > 0.5 else "low"

            print(f"[INFERENCE] Score: {score.item():.4f}, Mean_act: {mean_a_t:.4f}")

            # 5. SEMANTIC INJECTION & MSU
            g = packet_to_rdf(payload, risk=risk_label, score=score.item(), mean_a_t=mean_a_t)
            G += g

            packet_counter += 1
            if packet_counter % 100 == 0:
                send_to_fuseki(G)
                print("[GRAPH] MSU Sync executed")

            # 6. Ψ-GATING (Online Learning)
            if delta_H > tau_s and mean_a_t > theta:
                print(f"[Ψ GATE] Online update trigger (ΔH={delta_H:.3f}, a_t={mean_a_t:.3f})")
                arnn.train()
                optimizer.zero_grad()
                # On-the-fly heuristic for update
                y_target = torch.tensor([[1.0 if score.item() > 0.5 else 0.0]])
                L_tot = compute_hybrid_loss(score, y_target, arnn.W, W_gt, G)
                L_tot.backward()
                optimizer.step()
                arnn.eval()

            # Random Diagnostic
            if random.random() < 0.05:
                execute_diagnostic_queries()

            if risk_label == "high":
                print(f"🚨 [ALERT] HOST {payload['src']} UNDER ATTACK! Risk Score: {score.item():.4f}")
                apply_mitigation_policy(payload['src'], score.item())

    except Exception as e:
        print(f"[ERROR processing packet] {e}")

# ======================= FLASK & ENTRY POINT =======================
app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "fitted": fitted,
        "samples": len(X_train)
    })

@app.route('/metrics')
def metrics_endpoint():
    try:
        with open('/app/results/performance_report.json', 'r') as f:
            return jsonify(json.load(f))
    except:
        return jsonify({"error": "No metrics yet"})

@app.route('/plots')
def plots_endpoint():
    # Return list of available plots
    plots = {}
    for ws in ENTROPY_WINDOW_SIZES:
        if os.path.exists(f'/app/results/entropy_timeline_ws{ws}.png'):
            plots[f'ws_{ws}'] = 'available'
    return jsonify(plots)

def run_flask_server():
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('/app/results', exist_ok=True)
    os.makedirs('/app/results/sparql_queries', exist_ok=True)

    # 1. Init Entropy CSVs
    for ws in entropy_config['window_sizes']:
        with open(f'/app/results/entropy_analysis_ws{ws}.csv', 'w') as f:
            f.write("timestamp,window_size,entropy,delta_entropy,threshold,alarm\n")

    # 2. Init Mitigation CSV
    with open('/app/results/mitigation_times.csv', 'w') as f:
        f.write("timestamp,mitigation_time_ms,src_ip,risk_score\n")

    # 3. Init Detection CSV
    with open('/app/results/detection_times.csv', 'w') as f:
        f.write("timestamp,detection_time_ms\n")

    # 4. Init Query Performance JSONL
    with open('/app/results/query_performance.jsonl', 'w') as f:
        f.write("")

    # === (Startup Sequence) ===
    print("    Ψ-Risk-DT: DIGITAL TWIN STARTUP       ")
    print(f"[SYSTEM] Random Seed: {RANDOM_SEED}")
    print(f"[SYSTEM] Entropy Windows: {entropy_config['window_sizes']}")
    print(f"[SYSTEM] Ψ-Gating Threshold: {tau_s}")
    print(f"[SYSTEM] Baseline Duration: {BASELINE_DURATION}s")

    # 5. Start Flask
    print("[SYSTEM] Starting Health/Metrics Server on port 8080...")
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()

    # 6. Start MQTT
    client = mqtt.Client(client_id="digital-twin-psi", protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        print("[SYSTEM] Connecting to MQTT Broker...")
        client.connect("mqtt-broker", 1883)
        print("[SYSTEM] Connected. Waiting for IoT traffic...")
        client.loop_forever()
    except Exception as e:
        print(f"[FATAL] Cannot connect to MQTT: {e}")