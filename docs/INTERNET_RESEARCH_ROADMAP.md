# Internet Research Roadmap

This document captures concrete directions learned from broader internet research beyond the GitHub repositories already reviewed for this project.

## Current Themes Worth Building Toward

### 1. Heterogeneous Graphs
Transaction fraud rarely lives in a single node type. Users, devices, merchants, IPs, transactions, and accounts form a heterogeneous network.

Why this matters here:
- the current project already models several of these entities conceptually
- the next upgrade should represent them explicitly as typed nodes and typed edges

Primary source:
- PyTorch Geometric heterogeneous graph documentation:
  `https://pytorch-geometric.readthedocs.io/en/stable/notes/heterogeneous.html`

### 2. Temporal Graph Signals
Fraud often appears in bursts, sessions, and sequences rather than only static relationships.

Why this matters here:
- this repo already simulates burst timing
- the next step is turning time into first-class graph structure or event-stream data

Primary sources:
- PyTorch Geometric Temporal introduction:
  `https://pytorch-geometric-temporal.readthedocs.io/en/stable/notes/introduction.html`
- PyG `TemporalData` API:
  `https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/temporal.html`

### 3. Fraud Camouflage and Context
Recent fraud papers keep emphasizing that fraudsters blend in locally while behaving abnormally in context.

Why this matters here:
- this supports keeping graph context, degree, burst timing, and relation-aware features in the roadmap

Research example:
- `https://www.sciencedirect.com/science/article/pii/S0957417424023406`

### 4. Platform-Oriented Experimentation
The best public repos combine research readability with reproducible experiment structure.

Why this matters here:
- this project now includes config files, model catalog APIs, staged notebooks, and export scripts

## What Was Added to the Project Because of This Research

- `temporal_gnn`, `hetero_gnn`, and `fraudgt` model placeholders in the model catalog
- temporal and heterogeneous experiment config templates
- research-landscape and research-sources APIs
- docs that connect current implementation choices to future graph-learning directions
