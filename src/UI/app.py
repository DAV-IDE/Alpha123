import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import numpy as np

# Import logic from backend file
from psi_logic import initialize_psi_graph, highlight_nodes, causal_path, get_xai_explanation, \
    translate_nl_to_sparql_simulated, get_msu_subgraph_details

G_initial, tau_s, theta, highlighted_nodes_initial = initialize_psi_graph()
pos_initial = nx.spring_layout(G_initial, seed=42, k=6.0, iterations=100) # Use same seed for stable layout


def create_graph_figure(G, pos, tau_s, theta, traced_path=None):
    """
    Creates the complete Plotly figure based on the graph state G.
    Accepts 'traced_path' to highlight the causal path.
    """

    # Recalculate highlighted nodes based on current graph state (Dynamic Visual Gating)
    current_highlighted_nodes = highlight_nodes(G, tau_s, theta)

    # Initialization of lists for edges (untraced and traced)
    edge_x = []  # Standard edges (untraced)
    edge_y = []
    traced_edge_x = []  # Causal path edges (highlighted)
    traced_edge_y = []

    # Node positioning and color assignment (based on Gating result)
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_colors = ['red' if n in current_highlighted_nodes else 'blue' for n in G.nodes()]

    node_hover_info = []

    for n in G.nodes():
        # Base information (Risk State)
        text = f"<b>{n}</b>"
        text += f"<br>Entropy (ΔHt): {G.nodes[n]['entropy']:.4f}"
        text += f"<br>Activation (at): {G.nodes[n]['activation']:.4f}"

        # 1. Outgoing Links (Risk Propagation)
        # Lists all neighbors this node propagates risk to, along with edge weight.
        outgoing = [(neighbor, G.edges[n, neighbor]['weight']) for neighbor in G.successors(n)]
        if outgoing:
            text += "<br>--- OUTGOING LINKS (Propagates To) ---"
            for neighbor, weight in outgoing:
                text += f"<br>  → {neighbor} (W: {weight:.2f})"  # '→' indicates direction

        # 2. Incoming Links (Risk Source)
        # Lists all nodes that propagate risk into this node, along with edge weight.
        incoming = [(neighbor, G.edges[neighbor, n]['weight']) for neighbor in G.predecessors(n)]
        if incoming:
            text += "<br>--- INCOMING LINKS (Source Risk From) ---"
            for neighbor, weight in incoming:
                text += f"<br>  ← {neighbor} (W: {weight:.2f})"

        node_hover_info.append(text)

    # 1. Splitting Edges (Normal vs. Traced)
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        is_traced = False
        if traced_path:
            try:
                # Check if u and v are consecutive in the shortest path
                if traced_path.index(u) == traced_path.index(v) - 1:
                    is_traced = True
            except ValueError:
                pass

        if is_traced:
            traced_edge_x.extend([x0, x1, None])
            traced_edge_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    # 2. Trace 1: Standard Edges
    # Note: Weights are now visible only through node hover, not edge hover.
    edge_trace_normal = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none'
    )

    # 3. Trace 2: Traced Edges
    edge_trace_traced = go.Scatter(
        x=traced_edge_x, y=traced_edge_y, mode='lines',
        line=dict(width=3, color='#FFC300'),
        hoverinfo='none'
    )

    # 4. Trace 3: Nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        hovertext=node_hover_info,  # Use the rich hover data generated above
        hoverinfo='text',
        marker=dict(size=20, color=node_colors, line=dict(width=2)),
        text=list(G.nodes()), textposition="bottom center"
    )

    # 5. Return the Complete Figure
    return {
        'data': [edge_trace_normal, edge_trace_traced, node_trace],
        'layout': go.Layout(
            showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title=f'Neurosymbolic Graph (Threshold θ={theta:.4f})'
        )
    }


# --- Plotly Dash Layout ---
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'padding': '20px'}, children=[
    dcc.Store(id='graph-state', data=nx.node_link_data(G_initial)),

    html.H1("Ψ-Risk-DT: Neurosymbolic Interface",
            style={'textAlign': 'center', 'color': '#0A1A32'}),
    html.P(f"Entropy Threshold (τs): {tau_s}, Activation Threshold (θ): {theta:.4f}",
           style={'textAlign': 'center', 'fontSize': '1.1em'}),

    html.Div([
        # Graph Column
        html.Div(style={'width': '49%', 'display': 'inline-block'}, children=[
            dcc.Graph(id='psi-graph-visualization', figure=create_graph_figure(G_initial, pos_initial, tau_s, theta))
        ]),

        # Interaction Column
        html.Div(style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '0 10px'},
                 children=[

                     html.H3("1. Node Details & XAI", style={'color': '#007BFF'}),
                     html.P("Nodes to inspect (e.g., Node_1, Node_2, Node_3):"),
                     dcc.Input(id='nodes-inspect-input', type='text', placeholder='Node_1, Node_2',
                               style={'width': '100%', 'margin-bottom': '10px'}),
                     html.Button('Show XAI Details', id='inspect-button', n_clicks=0,
                                 style={'margin-bottom': '20px'}),
                     html.Div(id='details-output',
                              style={'border': '1px solid #ccc', 'padding': '10px', 'min-height': '100px',
                                     'white-space': 'pre-wrap', 'background-color': '#fff', 'margin-bottom': '20px', 'overflow-x': 'auto'}),

                     html.Hr(),

                     html.H3("2. NL Query", style={'color': '#007BFF'}),
                     dcc.Input(id='nl-query-input', type='text',
                               placeholder='show nodes with entropy > 0.3 AND activation > 0.6',
                               style={'width': '100%', 'margin-bottom': '10px'}),
                     html.Button('Execute Query', id='nl-query-button', n_clicks=0, style={'margin-bottom': '20px'}),
                     html.Div(id='nl-query-output',
                              style={'border': '1px solid #ccc', 'padding': '10px', 'white-space': 'pre-wrap',
                                     'margin-bottom': '20px', 'overflow-x': 'auto'}),

                     html.Hr(),

                     html.H3("3. Causal Threat Tracing", style={'color': '#007BFF', 'margin-top': '0'}),
                     dcc.Input(id='start-node-input', type='text',
                               placeholder='Start Node (e.g., Node_1)', style={'width': '45%', 'margin-right': '10px'}),
                     dcc.Input(id='end-node-input', type='text', placeholder='End Node (e.g., Node_20)',
                               style={'width': '45%', 'margin-bottom': '10px'}),
                     html.Button('Execute Causal Tracing', id='trace-button', n_clicks=0,
                                 style={'margin-bottom': '10px'}),
                     html.Div(id='causal-output',
                              style={'border': '1px solid #ccc', 'padding': '10px', 'white-space': 'pre-wrap',
                                     'margin-bottom': '20px', 'overflow-x': 'auto'}),

                     html.Hr(),

                     html.H3("4. What-If Analysis (Mitigation)", style={'color': '#007BFF'}),
                     dcc.Input(id='mit-source-input', type='text', placeholder='Source (e.g., Node_1)',
                               style={'width': '30%', 'margin-right': '5px'}),
                     dcc.Input(id='mit-target-input', type='text', placeholder='Target (e.g., Node_2)',
                               style={'width': '30%', 'margin-right': '5px'}),
                     dcc.Input(id='mit-weight-input', type='number', placeholder='New Weight (0.0 - 1.0)',
                               style={'width': '30%', 'margin-bottom': '10px'}),
                     html.Button('Simulate Mitigation', id='mitigate-button', n_clicks=0,
                                 style={'margin-bottom': '20px'}),
                     html.Div(id='mitigation-output',
                              style={'border': '1px solid #ccc', 'padding': '10px', 'white-space': 'pre-wrap',
                                     'overflow-x': 'auto'}),
                 ])
    ])
])


# --- CALLBACKS FOR FUNCTIONALITY ---

# Callback 1: Causal Tracing
@app.callback(
    [Output('causal-output', 'children'),
     Output('psi-graph-visualization', 'figure', allow_duplicate=True)],
    [Input('trace-button', 'n_clicks')],
    [State('start-node-input', 'value'), State('end-node-input', 'value'), State('graph-state', 'data')],
    prevent_initial_call=True
)
def update_causal_tracing(n_clicks, start, end, graph_data):
    if n_clicks == 0:
        return dash.no_update, dash.no_update

    G_current = nx.node_link_graph(graph_data)
    current_activations = np.array([G_current.nodes[n]['activation'] for n in G_current.nodes])
    current_theta = np.quantile(current_activations, 0.95)

    path, prob, details = causal_path(G_current, start, end)

    if prob > 0:
        # Create the new figure, passing the found path for highlighting
        new_figure = create_graph_figure(G_current, pos_initial, tau_s, current_theta, traced_path=path)

        output = [
            f"--- CAUSAL TRACING RESULT ---",
            f"Causal Path: {' -> '.join(path)}",
            f"PROPAGATION PROBABILITY (P_threat): {prob:.4f}",
            f"\nSegmented Contribution Explanation:"
        ]

        for d in details:
            output.append(f"  - Edge {d['source']} -> {d['target']}: Contribution (Weight) = {d['weight']:.2f}")

        # Add XAI explanation of the final node
        xai_exp = get_xai_explanation(G_current, end, tau_s, current_theta)
        output.append(f"\nXAI Explanation for {end}: {xai_exp}")

        # Return the textual output and the new figure
        return html.Pre('\n'.join(output)), new_figure

    # If no path is found, return the error message and NO FIGURE UPDATE
    return html.Pre(
        f"TRACING RESULT: {' '.join(details) if details else 'No path found or invalid nodes.'}"), dash.no_update


# Callback 2: Multi-Node Navigation
@app.callback(
    Output('details-output', 'children'),
    [Input('inspect-button', 'n_clicks')],
    [State('nodes-inspect-input', 'value'), State('graph-state', 'data')]
)
def update_node_details(n_clicks, nodes_input, graph_data):
    if n_clicks == 0 or not nodes_input:
        return dash.no_update

    G_current = nx.node_link_graph(graph_data)
    nodes_to_view = [n.strip() for n in nodes_input.split(',') if n.strip()]

    current_activations = np.array([G_current.nodes[n]['activation'] for n in G_current.nodes])
    current_theta = np.quantile(current_activations, 0.95)
    current_highlighted_nodes = highlight_nodes(G_current, tau_s, current_theta)

    output = []
    for node_name in nodes_to_view:
        if node_name in G_current.nodes:
            xai_exp = get_xai_explanation(G_current, node_name, tau_s, current_theta)
            status = 'ANOMALY (RED)' if node_name in current_highlighted_nodes else 'NORMAL'

            # Format Output
            output.append(f"--- [NODE DETAILS: {node_name}] ---")
            output.append(f"  > Detection Status: {status}")
            output.append(f"  > Entropy Attribute (ΔHt): {G_current.nodes[node_name]['entropy']:.4f}")
            output.append(f"  > ARNN Activation (at): {G_current.nodes[node_name]['activation']:.4f}")
            output.append(f"  > XAI Explanation (Semantic Tripla): {xai_exp}")
            output.append("-" * 30)
        else:
            output.append(f"\nError: Node '{node_name}' not found in graph.")

    # Add MSU Subgraph Partition Simulation
    if nodes_to_view and nodes_to_view[0] in G_current.nodes:
        output.append("\n\n--- MSU SUBGRAPH PARTITION SIMULATION (Nearby Assets) ---")
        msu_details = get_msu_subgraph_details(G_current, nodes_to_view)
        output.extend(msu_details)

    return html.Pre('\n'.join(output))


# Callback 3: Advanced NL Query
@app.callback(
    Output('nl-query-output', 'children'),
    [Input('nl-query-button', 'n_clicks')],
    [State('nl-query-input', 'value'), State('graph-state', 'data')]
)
def handle_nl_query(n_clicks, nl_query, graph_data):
    if n_clicks == 0 or not nl_query:
        return dash.no_update

    G_current = nx.node_link_graph(graph_data)

    sparql_query, nl_result = translate_nl_to_sparql_simulated(G_current, nl_query)

    if sparql_query:
        output = [
            f"--- NL QUERY RESULT (SPARQL Simulation) ---",
            f"NL Query Received: '{nl_query}'",
            f"\n--- GENERATED SPARQL QUERY ---",
            sparql_query, # print the full sparql string
            f"\nNodes satisfying the criteria (Simulation Result): {nl_result}"
        ]
        return html.Pre('\n'.join(output))

    return html.Pre(f"Query not recognized or no nodes found. (Expected format: show nodes with attr > X)")


# Callback 4: What-If Analysis
@app.callback(
    [Output('mitigation-output', 'children'),
     Output('graph-state', 'data'),
     Output('psi-graph-visualization', 'figure', allow_duplicate=True)],
    [Input('mitigate-button', 'n_clicks')],
    [State('mit-source-input', 'value'),
     State('mit-target-input', 'value'),
     State('mit-weight-input', 'value'),
     State('graph-state', 'data'),
     State('start-node-input', 'value'),
     State('end-node-input', 'value')],
    prevent_initial_call=True
)
def handle_what_if(n_clicks, s, t, new_weight, graph_data, start_trace_input, end_trace_input):
    if n_clicks == 0 or new_weight is None:
        return dash.no_update, dash.no_update, dash.no_update

    G_current = nx.node_link_graph(graph_data)

    # 1. Validation and Update
    if s not in G_current.nodes or t not in G_current.nodes or not G_current.has_edge(s, t):
        return html.Pre(f"Error: Edge {s} -> {t} invalid or nonexistent."), dash.no_update, dash.no_update

    old_weight = G_current.edges[s, t]['weight']

    try:
        new_weight = float(new_weight)
        if not (0 <= new_weight <= 1):
            return html.Pre("Error: Weight must be a number between 0 and 1."), dash.no_update, dash.no_update
    except ValueError:
        return html.Pre("Error: Weight must be a valid number."), dash.no_update, dash.no_update

    # 2. Update the Weight (Mitigation Simulation)
    G_current.edges[s, t]['weight'] = new_weight

    # 3. Recalculate Risk
    start_trace = start_trace_input
    end_trace = end_trace_input

    current_activations = np.array([G_current.nodes[n]['activation'] for n in G_current.nodes])
    current_theta = np.quantile(current_activations, 0.95)

    _, new_prob, _ = causal_path(G_current, start_trace, end_trace)

    # 4. Output and State Return
    output = [f"--- WHAT-IF RESULT ---"]
    output.append(f"Mitigation applied to edge {s} -> {t}.")
    output.append(f"Previous Weight: {old_weight:.2f} -> New Weight: {new_weight:.2f}")
    output.append(f"Recalculated P_threat ({start_trace} -> {end_trace}) is now: {new_prob:.4f}")

    # 5. Return the new graph state and figure
    new_graph_data = nx.node_link_data(G_current)
    new_figure = create_graph_figure(G_current, pos_initial, tau_s, current_theta, traced_path=None)

    return html.Pre('\n'.join(output)), new_graph_data, new_figure


# --- Application Launch ---
if __name__ == '__main__':
    # Run in terminal: python app.py
    app.run(debug=True)