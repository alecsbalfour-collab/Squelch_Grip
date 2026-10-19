# AGENTS.md — SquelchGrip Backend (`squelch_grip_backend`)

This file defines the architectural rules, coding conventions, and verification commands for all coding agents (including Argon / Devin and SWE-2) working in `D:\SquelchGrip\squelch_grip_backend`.

---

## 1. Workspace & Directory Layout

- **Workspace Root**: `D:\SquelchGrip\squelch_grip_backend`
- **Important Path Rule**: `D:\SquelchGrip\squelch_grip_backend` **is** the root backend package. Do **not** create a nested `squelch_grip/` subfolder inside `squelch_grip_backend/`.
- **Directory Structure**:
  ```text
  D:\SquelchGrip\squelch_grip_backend/
  ├── schemas/
  │   ├── __init__.py
  │   └── session_models.py       # Pydantic v2 domain & directive models
  ├── core/
  │   ├── __init__.py
  │   ├── context_graph.py        # ContextGraphOrchestrator (non-linear topic tree & barge-in)
  │   └── state_manager.py        # SessionStateManager (room & VoiceProfile registry)
  ├── tests/
  │   ├── __init__.py
  │   └── test_context_graph.py   # Async pytest suite
  ├── pyproject.toml              # Pytest & package configuration
  ├── requirements.txt
  ├── README.md
  └── AGENTS.md
  ```

---

## 2. Environment & Verification Commands

- **Python Runtime**: Python 3.14 (`C:\Users\Alec\AppData\Local\Python\pythoncore-3.14-64\python.exe`)
- **Run Unit Tests** (always use `python -m pytest` because the Python `Scripts` directory is not on `PATH`):
  ```bash
  python -m pytest -v
  ```
- **Install Dependencies**:
  ```bash
  python -m pip install -r requirements.txt
  ```
- **Pytest Async Configuration**: Configured in `pyproject.toml` with `asyncio_mode = "auto"` and `pythonpath = ["."]`.

---

## 3. Code & Schema Conventions

1. **Pydantic v2 Strictness**:
   - All models in `schemas/` must inherit from `pydantic.BaseModel` and set `model_config = ConfigDict(extra="forbid")`.
   - Use `Field(default_factory=...)` for mutable defaults (`list`, `dict`, timestamps, UUIDs).
   - All enums must inherit from `(str, Enum)` so they serialize cleanly to JSON and compare equal to raw strings when needed.
2. **Backwards-Compatible Command Payloads**:
   - `BargeInDirective` is a strongly typed Pydantic model returned by `ContextGraphOrchestrator.handle_barge_in()`, and also implements `__getitem__` and `.get()` so callers using either `directive.command` or `directive["command"]` work without breaking.
3. **Concurrency & State Mutation**:
   - Mutating methods on `ContextGraphOrchestrator` and `SessionStateManager` must remain `async` and guard state transitions with `async with self._lock:` (`asyncio.Lock`).
4. **Interruption Semantics**:
   - `InterruptionType.HARD_INTERRUPT`: Sets `is_ai_speaking = False`, records utterance, emits `AudioCommandType.HALT` (`HALT_AUDIO`).
   - `InterruptionType.BACKCHANNEL`: Preserves `is_ai_speaking`, logs backchannel utterance on `current_node`, emits `AudioCommandType.CONTINUE` (`LOG_BACKCHANNEL`).
   - `InterruptionType.BACKGROUND_EVENT`: Preserves `is_ai_speaking`, does not add noise source to `active_speakers` or node utterances, emits `AudioCommandType.IGNORE` (`SQUELCH_BACKGROUND`).
5. **Module Exports**:
   - Whenever adding new schemas to `schemas/` or new classes/functions to `core/`, export them in `schemas/__init__.py` or `core/__init__.py` (`__all__`).
6. **Mandatory Verification**:
   - Never mark a task complete without running `python -m pytest -v` and confirming 100% of existing and new tests pass.
