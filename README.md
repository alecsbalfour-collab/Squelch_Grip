# SquelchGrip Backend (`squelch_grip_backend`) — v1.0-alpha

Foundational core engine for **SquelchGrip**, a full-duplex, multi-speaker voice management and non-linear conversational context graph orchestrator.

## Repository Structure

```text
squelch_grip_backend/
├── api/
│   ├── __init__.py
│   └── server.py
├── schemas/
│   ├── __init__.py
│   ├── session_models.py
│   ├── lexicon_models.py
│   ├── turn_models.py
│   ├── prompt_models.py
│   └── navigation_models.py
├── core/
│   ├── __init__.py
│   ├── state_manager.py
│   ├── context_graph.py
│   ├── voice_biometrics.py
│   ├── interruption_classifier.py
│   ├── lexicon_manager.py
│   ├── navigation_detector.py
│   ├── turn_pipeline.py
│   └── prompt_compiler.py
├── tests/
│   ├── __init__.py
│   ├── test_context_graph.py
│   ├── test_biometrics_and_classifier.py
│   ├── test_lexicon_manager.py
│   ├── test_turn_pipeline.py
│   ├── test_prompt_compiler.py
│   ├── test_navigation_detector.py
│   └── test_api_server.py
├── pyproject.toml
├── requirements.txt
├── AGENTS.md
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
