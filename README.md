# Ψ-Risk-DT: Supplementary Materials

This repository contains supplementary materials for:

> Pazzi, R., Facheris, D., Tosi, D. (2026). "Ψ-Risk-DT: A Neurosymbolic
> Framework for Adaptive Zero-Day Threat Detection in Digital Twin Ecosystems."

## Validation Protocol and Scope of Claims
The empirical results presented in the paper were generated via a **fully reproducible simulation** focused on a controlled Network Time Protocol (NTP) Amplification scenario. This validation was specifically designed to test the core capabilities of the entropy-gated coupling ($\Psi$) under volumetric attack conditions.

It is essential to note that this initial validation constitutes a Proof of Concept (PoC). The quantitative claims regarding broader Zero-Day resilience and reduced False Positive Rates are based on the theoretical extrapolation achieved in controlled settings, with full generalization pending future work on Out-of-Distribution (OOD) benchmarks (e.g., Kitsune 2019, WiseML 2024).

## Contents

- `additions/` — This folder contains the mathematical and theoretical refinements for Ψ-Risk-DT. It formally defines the Hybrid Loss Function and the semantic distance metric, providing the rigorous proof of concept for how the neurosymbolic operator Ψ achieves alignment between neural activations and RDF graph semantics.
- `src/` — Implementation source code and PoC Ψ-UI .

## User Interface (UI) & Architectural Diagrams
*   **Pipeline Architecture:** (Figure 1): `supplementary/figure1.pdf`
*   **MSU Schema:** (Figure 2): `supplementary/figure2.pdf`
*   **Explainer UI Prototype:** Code and documentation for the Ψ-UI are provided in the `src/UI` folder, demonstrating the framework's explainability capability. **(Note: The UI is a visualization tool and does not constitute part of the formal empirical validation reported in the paper.)**
