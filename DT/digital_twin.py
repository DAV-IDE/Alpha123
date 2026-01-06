"""
DIGITAL TWIN - Complete Pipeline Implementation O1-O6
Integrato con pseudocodice Ψ-Risk-DT: ARNN ricorrente, hybrid loss, gating entropico.
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
def fuseki_auth():
    user = os.getenv("FUSEKI_USER")
    pwd = os.getenv("FUSEKI_PASSWORD")
    if user and pwd:
        return HTTPBasicAuth(user, pwd)
    return None

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
packet_counter = 0  # Per sync periodico

# Neurosymbolic params
tau_s = 0.15  # Entropy threshold
theta = 0.7   # Activation threshold
alpha, beta, gamma = 0.4, 0.3, 0.3  # Hybrid loss weights
input_size = 12  # Da vectorize
hidden_size = 64
prev_a = torch.zeros(hidden_size)  # Stato iniziale ARNN
G = Graph()  # Graph locale per L_sem (MSU-like)
W_gt = torch.randn(hidden_size, hidden_size)  # Placeholder teacher weights

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
        g.add((pkt_uri, NTW.risk_activation, Literal(mean_a_t, datatype=XSD.float)))  # Ψ inject

    return g

def send_to_fuseki(graph):
    # Serializzo in N-Triples (va bene dentro INSERT DATA)
    triples_nt = graph.serialize(format="nt")

    update = f"INSERT DATA {{ {triples_nt} }}"

    try:
        response = requests.post(
            "http://fuseki:3030/ds/update",
            data=update,
            headers={"Content-Type": "application/sparql-update"},
            auth=fuseki_auth(),   # <-- QUI la differenza
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

# ======================= O4 - ARNN CORE (Integrato con pseudocodice) =======================
class ARNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # Concat hidden + associated
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, prev_a):
        _, hn = self.rnn(x.unsqueeze(0))  # hn shape: (1, hidden_size)
        hn = hn.squeeze(0)
        associated = torch.matmul(prev_a, self.W) + self.b
        combined = torch.cat((hn, associated), dim=0)
        out = torch.sigmoid(self.fc(self.dropout(combined)))
        return out, hn  # Return score e new state

arnn = ARNN(input_size=input_size, hidden_size=hidden_size)
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

def compute_hybrid_loss(score, y_true, W, W_gt, G, alpha=alpha, beta=beta, gamma=gamma):
    # L_cls: BCE (scalar)
    L_cls = criterion(score, y_true)

    # L_graph: Frobenius norm
    L_graph = torch.norm(W - W_gt, p='fro') ** 2

    # L_sem: Media risk_activation dal graph locale vs score
    risk_activations = []
    for s, p, o in G.triples((None, NTW.risk_activation, None)):
        risk_activations.append(float(o))
    embeddings_graph = np.mean(risk_activations) if risk_activations else 0.0
    L_sem = (score.item() - embeddings_graph) ** 2

    L_total = alpha * L_cls + beta * L_graph + gamma * L_sem
    return L_total

# Altre funzioni (calculate_confidence_interval, calculate_performance_metrics, save_performance_report, generate_plots) rimangono invariate...

def train_model_async(X_train, y_train, normal_samples, attack_samples):
    def training_thread():
        global fitted, arnn, criterion, optimizer, performance_metrics, training_in_progress, prev_a

        try:
            performance_metrics['training_start_time'] = time.time()
            print(f"[ARNN] Starting training with {len(X_train)} samples")

            total = normal_samples + attack_samples
            weight_for_0 = total / (2 * normal_samples) if normal_samples > 0 else 1
            weight_for_1 = total / (2 * attack_samples) if attack_samples > 0 else 1

            print(f"[ARNN] Class weights - Normal: {weight_for_0:.2f}, Attack: {weight_for_1:.2f}")

            # Processa sequenzialmente per stato ricorrente
            arnn.train()
            local_state = torch.zeros(hidden_size)  # Stato locale per training
            for epoch in range(200):
                epoch_loss = 0
                for i in range(len(X_train)):
                    optimizer.zero_grad()
                    x_tensor = torch.tensor(X_train[i], dtype=torch.float32).unsqueeze(0)
                    y_tensor = torch.tensor([[y_train[i]]], dtype=torch.float32)
                    score, new_state = arnn(x_tensor, local_state)
                    local_state = new_state.detach()

                    # Hybrid loss
                    L_total = compute_hybrid_loss(score, y_tensor, arnn.W, W_gt, G)

                    L_total.backward()
                    optimizer.step()
                    epoch_loss += L_total.item()

                if epoch % 40 == 0:
                    print(f"[ARNN] Epoch {epoch}, Avg Loss: {epoch_loss / len(X_train):.6f}")

            arnn.eval()
            with torch.no_grad():
                train_scores = []
                local_state = torch.zeros(hidden_size)
                for x in X_train:
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    score, new_state = arnn(x_tensor, local_state)
                    local_state = new_state
                    train_scores.append(score.item())

                train_predictions = [1 if s > 0.5 else 0 for s in train_scores]
                accuracy = sum(p == y for p, y in zip(train_predictions, y_train)) / len(y_train)
                print(f"[ARNN] Training completed. Accuracy: {accuracy:.4f}")

            fitted = True
            training_in_progress = False
            print(f"[ARNN] MODEL TRAINING COMPLETED! Switching to inference mode.")

            y_true = np.array(y_train)
            y_pred = np.array(train_predictions)
            y_scores = np.array(train_scores)

            save_performance_report(y_true, y_pred, y_scores)
            generate_plots()

        except Exception as e:
            print(f"[TRAINING ERROR] {e}")
            training_in_progress = False

    thread = threading.Thread(target=training_thread, daemon=True)
    thread.start()

# Altre funzioni (run_sparql_query, execute_diagnostic_queries, app routes, run_health_server, apply_mitigation_policy, on_connect) rimangono invariate...

def on_message(client, userdata, message):
    global fitted, training_mode, training_samples, arnn, criterion, optimizer
    global performance_metrics, training_in_progress, entropy_config, prev_a, G, packet_counter

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
        y_true_heuristic = torch.tensor([[1.0 if payload["size"] > 300 else 0.0]])  # Per hybrid loss online

        if not fitted:
            # Heuristic labeling (come originale)
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
            G += g  # Aggiungi a locale
            send_to_fuseki(g)

        else:
            # O4 - INFERENCE PHASE (con stato ricorrente)
            inference_start = time.time()

            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                score, new_a = arnn(x_tensor, prev_a)
            prev_a = new_a  # Aggiorna stato globale

            mean_a_t = torch.mean(new_a).item()  # Per gating e inject

            inference_time = (time.time() - inference_start) * 1000
            performance_metrics['inference_times'].append(inference_time)

            print(f"[INFERENCE] Prediction score: {score.item():.6f}, Mean_a_t: {mean_a_t:.6f}, Time: {inference_time:.2f}ms")

            risk_label = "high" if score > 0.5 else "low"

            # Writing detection time to CSV
            with open('/app/results/detection_times.csv', 'a') as f:
                f.write(f"{payload['ts']},{inference_time}\n")

            # O5 - SEMANTIC GRAPH INJECTION (con Ψ inject)
            g = packet_to_rdf(payload, risk=risk_label, score=score.item(), mean_a_t=mean_a_t)
            G += g  # Modular update locale (MSU)
            send_to_fuseki(g)

            packet_counter += 1
            if packet_counter % 100 == 0:
                send_to_fuseki(G)  # Sync periodico per efficienza
                print("[GRAPH] Synced local graph to Fuseki")

            # Gating per update online (pseudocodice integration)
            if delta_H > tau_s and mean_a_t > theta:
                print("[Ψ GATE] Triggering online update")
                arnn.train()
                optimizer.zero_grad()
                L_total = compute_hybrid_loss(score, y_true_heuristic, arnn.W, W_gt, G)
                L_total.backward()
                optimizer.step()
                arnn.eval()

            # O6 - DYNAMIC UPDATE LOOP (Diagnostic Queries)
            if random.random() < 0.1:
                execute_diagnostic_queries()

            if risk_label == "high":
                print(f"[ALERT] Packet {payload['device']} HIGH RISK: {score.item():.3f}")
                apply_mitigation_policy(payload['src'], score.item())

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

def run_health_server(host="0.0.0.0", port=8080):
    """
    Minimal health endpoint to keep the container alive and provide a readiness probe.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/health", "/healthz", "/ready", "/"):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"OK\n")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # silence default HTTP server logs
            return

    HTTPServer((host, port), Handler).serve_forever()


    # Start Flask Health Server (separate thread)
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    # Start MQTT client
    client = mqtt.Client(client_id="digital-twin", protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("mqtt-broker", 1883)
    client.loop_forever()