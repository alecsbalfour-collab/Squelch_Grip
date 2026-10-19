# SquelchGrip Backend (`squelch_grip_backend`) — v1.0-alpha

Foundational core engine for **SquelchGrip**, a full-duplex, multi-speaker voice management and non-linear conversational context graph orchestrator.

## Repository Structure

```text
squelch_grip_backend/
├── schemas/
│   ├── __init__.py
│   └── session_models.py
├── core/
│   ├── __init__.py
│   ├── state_manager.py
│   └── context_graph.py
├── tests/
│   ├── __init__.py
│   └── test_context_graph.py
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest
```
