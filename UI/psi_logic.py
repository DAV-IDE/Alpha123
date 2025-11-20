import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import re
import random


# --- CONFIGURATION PARAMETERS ---
NUM_NODES = 35 # Target number of nodes for the scaled simulation
EDGE_PROBABILITY = 0.08  # Probability for creating an edge (Erdos-Renyi model)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


class SimpleARNN(nn.Module):
    """Simulates the Associated Random Neural Network (ARNN)."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out_mapped = self.linear(out)
        # Use the output from the linear layer, adapted for sequential coherence
        return torch.sigmoid(out_mapped).squeeze(0).squeeze(-1)


# --- ANALYSIS AND GATING FUNCTIONS ---

def highlight_nodes(G, tau_s, theta):
    """Implements Dynamic Visual Gating based on Entropy (tau_s) OR Activation (theta)."""
    return [n for n in G.nodes if G.nodes[n]['entropy'] > tau_s or G.nodes[n]['activation'] > theta]


def translate_nl_to_sparql_simulated(G, nl_query):
    """
    Translates an NL query into the required SPARQL format, handling AND, OR, and simple filters.
    (Simulates the output of a Transformer-based NLP module).
    """

    # Mapping NL keywords to graph predicates
    MAPPINGS = {
        'entropy': 'hasEntropyDeviation',
        'activation': 'hasARNNActivation',
        'sensor': 'DTComponent_Sensor'
    }

    # Patterns must be checked in order of complexity (OR, AND, Simple)

    # 1. Pattern for OR
    pattern_or = r'show nodes with (\w+) > (\d+\.\d*) OR (\w+) > (\d+\.\d*)'
    match_or = re.search(pattern_or, nl_query, re.IGNORECASE)

    # 2. Pattern for AND
    pattern_and = r'show nodes with (\w+) > (\d+\.\d*) AND (\w+) > (\d+\.\d*)'
    match_and = re.search(pattern_and, nl_query, re.IGNORECASE)

    # 3. Pattern for simple
    pattern_simple = r'show nodes with (\w+) > (\d+\.\d*)'
    match_simple = re.search(pattern_simple, nl_query, re.IGNORECASE)

    # --- TRANSLATION LOGIC ---

    if match_or:
        attr1, val1_str, attr2, val2_str = match_or.groups()
        val1, val2 = float(val1_str), float(val2_str)

        predicato1 = MAPPINGS.get(attr1.lower(), attr1.lower())
        predicato2 = MAPPINGS.get(attr2.lower(), attr2.lower())

        # Construct SPARQL string with ||
        sparql_query = (
            f"SELECT ?entity WHERE {{\n"
            f"  ?entity :{predicato1} ?value1 .\n"
            f"  ?entity :{predicato2} ?value2 .\n"
            f"  FILTER (?value1 > {val1} || ?value2 > {val2})\n"
            f"}}"
        )
        # Python Simulation Execution (OR Logic)
        result = [n for n in G.nodes if
                  G.nodes[n].get(attr1.lower()) is not None and G.nodes[n][attr1.lower()] > val1 or
                  G.nodes[n].get(attr2.lower()) is not None and G.nodes[n][attr2.lower()] > val2]

        return sparql_query, result

    elif match_and:
        attr1, val1_str, attr2, val2_str = match_and.groups()
        val1, val2 = float(val1_str), float(val2_str)

        predicato1 = MAPPINGS.get(attr1.lower(), attr1.lower())
        predicato2 = MAPPINGS.get(attr2.lower(), attr2.lower())

        # Construct SPARQL string with &&
        sparql_query = (
            f"SELECT ?entity WHERE {{\n"
            f"  ?entity :{predicato1} ?value1 .\n"
            f"  ?entity :{predicato2} ?value2 .\n"
            f"  FILTER (?value1 > {val1} && ?value2 > {val2})\n"
            f"}}"
        )

        # Python Simulation Execution (AND Logic)
        result = [n for n in G.nodes if
                  G.nodes[n].get(attr1.lower()) is not None and G.nodes[n][attr1.lower()] > val1 and
                  G.nodes[n].get(attr2.lower()) is not None and G.nodes[n][attr2.lower()] > val2]

        return sparql_query, result

    elif match_simple:
        nl_attr, val_str = match_simple.groups()
        value = float(val_str)

        predicato = MAPPINGS.get(nl_attr.lower(), nl_attr.lower())

        sparql_query = (
            f"SELECT ?entity WHERE {{\n"
            f"  ?entity :{predicato} ?value .\n"
            f"  FILTER (?value > {value})\n"
            f"}}"
        )

        result = [n for n in G.nodes if
                  G.nodes[n].get(nl_attr.lower()) is not None and G.nodes[n][nl_attr.lower()] > value]

        return sparql_query, result

    # Failure
    return None, []


def causal_path(G, start, end):
    """Causal Threat Tracing. Returns path and probability."""
    try:
        path = nx.shortest_path(G, start, end)
        path_details = []
        prob = 1.0

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            weight = G.edges[source, target]['weight']
            prob *= weight
            path_details.append({
                'source': source,
                'target': target,
                'weight': weight,
            })

        return path, prob, path_details
    except nx.NetworkXNoPath:
        return [], 0.0, [f"Error: No path found between {start} and {end}."]
    except nx.NodeNotFound:
        return [f"Error: Node not found"], 0.0, []


def get_xai_explanation(G, node, tau_s, theta):
    """Generates a specific XAI explanation based on the Gating condition."""
    entropy = G.nodes[node]['entropy']
    activation = G.nodes[node]['activation']

    is_high_entropy = entropy > tau_s
    is_high_activation = activation > theta

    explanation = ""
    if is_high_entropy and is_high_activation:
        explanation = f"Anomaly Detected: Dual Signal (Entropy={entropy:.4f} > {tau_s} AND Activation={activation:.4f} > {theta}). Tripla: :hasEntropySpike AND :hasHighARNNActivation"
    elif is_high_entropy:
        explanation = f"Anomaly Detected: Statistical Risk (Entropy={entropy:.4f} > {tau_s}). Tripla: :hasEntropySpike"
    elif is_high_activation:
        explanation = f"Anomaly Detected: ARNN Propagated Risk (Activation={activation:.4f} > {theta}). Tripla: :hasHighARNNPropagation"
    else:
        explanation = "NORMAL Node. No alarm threshold exceeded."

    return explanation


def get_msu_subgraph_details(G, central_nodes):
    """
    Simulates Modular Semantic Update (MSU) partitioning.
    Returns details about the local subgraph formed by central_nodes and their immediate neighbors.
    """
    details = []

    for central_node in central_nodes:
        if central_node not in G.nodes:
            details.append(f"Error: Central node '{central_node}' not found.")
            continue

        # Get neighbors: nodes connected to the central node (in and out)
        neighbors = set(G.predecessors(central_node)).union(set(G.successors(central_node)))
        local_nodes = list(neighbors.union({central_node}))

        # Extract the local subgraph (G_t(i))
        subgraph = G.subgraph(local_nodes)

        # Simulate efficiency annotation
        efficiency = f"{len(subgraph.nodes)}/{len(G.nodes)}"

        details.append(f"--- MSU LOCAL PARTITION: {central_node} (Efficiency: {efficiency}) ---")
        details.append(f"  Local Nodes: {local_nodes}")
        details.append(f"  Local Edges (Semantic Links): {len(subgraph.edges)}")

        # Simulate loading the local edges and their weights
        for u, v, data in subgraph.edges(data=True):
            details.append(f"    - Link {u} -> {v}: Weight={data['weight']:.2f}")

    return details


def initialize_psi_graph():
    """Creates and initializes the neurosymbolic graph G."""

    # Create nodes: 'Node_1', 'Node_2', ..., 'Node_35'
    nodes = [f"Node_{i + 1}" for i in range(NUM_NODES)]

    # Generate a random graph structure
    G = nx.fast_gnp_random_graph(n=NUM_NODES, p=EDGE_PROBABILITY, directed=True, seed=SEED)
    G = nx.relabel_nodes(G, {i: nodes[i] for i in range(NUM_NODES)})

    for node in G.nodes():
        # Entropies: Randomly distributed (0.0 to 0.6)
        G.add_node(node, entropy=np.random.uniform(0.0, 0.6), activation=0.0)

    # Initialize edge weights (propagation probability)
    for u, v in G.edges():
        # Weights: Randomly distributed (0.2 to 0.9)
        G.edges[u, v]['weight'] = np.random.uniform(0.2, 0.9)

    entropy_values = np.array([G.nodes[n]['entropy'] for n in G.nodes]).reshape(1, -1, 1)
    arnn = SimpleARNN(1, 1)

    # Execute inference for all nodes
    activations = arnn(torch.tensor(entropy_values, dtype=torch.float)).detach().numpy().flatten()

    # If the ARNN output is non-deterministic (size mismatch), truncate or pad for safety
    if len(activations) != len(nodes):
        activations = activations[:len(nodes)]

    # Semantic Injection
    for i, node in enumerate(G.nodes()):
        if i < len(activations):
            G.nodes[node]['activation'] = activations[i]

    tau_s = 0.2
    theta = np.quantile(activations, 0.95)
    highlighted_nodes = highlight_nodes(G, tau_s, theta)

    return G, tau_s, theta, highlighted_nodes