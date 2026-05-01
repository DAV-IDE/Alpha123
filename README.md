# Ψ-Risk-DT: Supplementary Materials

This repository contains supplementary materials for:

> Pazzi, R., Facheris, D., Tosi, D. (2026). "Ψ-Risk-DT: A Neurosymbolic
> Framework for Adaptive Zero-Day Threat Detection in Digital Twin Ecosystems."

## Validation Protocol and Scope of Claims
The empirical results presented in the paper were generated via a **fully reproducible simulation** focused on a controlled Network Time Protocol (NTP) Amplification scenario. This validation was specifically designed to test the core capabilities of the entropy-gated coupling ($\Psi$) under volumetric attack conditions.

The numerical results reported in the paper (AUC, FPR, latency, MSU) are directly measured on the NTP-amplification scenario. Their generalization to broader Zero-Day classes is positional and depends on future work on OOD benchmarks such as Kitsune, WiseML 2024 and Sec4ML 2023.

## Contents

- `additions/` — This folder contains the mathematical and theoretical refinements for Ψ-Risk-DT. It formally defines the Hybrid Loss Function and the semantic distance metric, providing the rigorous proof of concept for how the neurosymbolic operator Ψ achieves alignment between neural activations and RDF graph semantics.
- `src/DT/` — core neurosymbolic Digital Twin pipeline, ARNN, hybrid loss, entropy gating, MQTT ingestion, SPARQL injection, REST endpoints
- `src/UI/` — Dash/Plotly proof-of-concept visualizer, solo illustrativo
    
## User Interface (UI) & Architectural Diagrams
*   **Pipeline Architecture:** (Figure 1): `supplementary/figure1.pdf`
*   **MSU Schema:** (Figure 2): `supplementary/figure2.pdf`
*   **Explainer UI Prototype:** Code and documentation for the Ψ-UI are provided in the `src/UI` folder, demonstrating the framework's explainability capability. **(Note: The UI is a visualization tool and does not constitute part of the formal empirical validation reported in the paper.)**
