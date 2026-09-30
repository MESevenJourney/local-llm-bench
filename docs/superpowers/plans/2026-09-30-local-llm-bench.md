# Local LLM Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a model- and engine-extensible local LLM benchmark project with one-command setup/run, reproducible scenarios, and traceable, sanitized results.

**Architecture:** A Python standard-library core loads TOML profiles and JSON scenarios, sends streaming OpenAI-compatible requests, normalizes measurements, and writes run manifests and reports. Engine adapters own environment setup, service lifecycle, and draft verification. Initial profiles cover Qwen3.8-27B and Bonsai 2 on TensorFold, plus Qwen3.8-27B on isolated oMLX.

**Tech Stack:** Python 3.13 for pinned engine environments; Python 3.11+ for the core; standard library for the runner and tests; shell entrypoint; TOML/JSON profile data.

**Spec:** `docs/superpowers/specs/2026-09-30-local-llm-bench-design.md`

## Global Constraints

- The first release targets Apple Silicon macOS; model and engine profiles remain independent and extensible.
- `./bench.sh run --profile PROFILE --scenario SCENARIO` bootstraps the selected software environment, validates local model files, starts and verifies the engine, runs the test, stops only its owned process, and writes a report.
- Model weights are never committed or downloaded silently. Download is an explicit opt-in; a missing or unpinned model revision is recorded as unknown rather than invented.
- A speculative profile must confirm the configured drafter is attached and used; silent fallback invalidates the run.
- Each run records project commit/dirty state, profile and scenario hashes, hardware/OS, runtime versions, model/drafter identifiers and fingerprints, parameters, per-turn JSON/CSV metrics, and sanitized logs.
- Shareable output excludes API keys, private absolute paths, private conversations, model weights, and hidden reasoning by default.
- Cross-engine decode uses one client-side streaming formula; server-reported speeds remain separate and missing values stay null.
- Natural multi-turn and frozen-history replay are separate test modes. Reports state sample count, run variation, and actual longest prompt.
- Existing oMLX personal settings and services are not modified; benchmark oMLX uses an isolated base directory and port.
- Documentation is Chinese-first. The X Article remains a Qwen case study and does not claim the project is TensorFold- or oMLX-maintained.

## Review Focus

- Missing or malformed SSE usage / finish reason — reject an incomplete run; test truncated streams and missing fields in Task 2.
- A drafter requested but not loaded — refuse to mark the profile valid; test a startup log with fallback text in Task 3.
- Model revision absent in a local cache — preserve `unknown` plus config/tokenizer fingerprints; test in Task 1.
- Secret or user path in a manifest/export — redact before writing shareable output; test in Task 1 and Task 2.
- oMLX profile points at the daily `~/.omlx` tree — refuse before writing config or starting the server; test in Task 3.

---

### Task 1: Profiles, scenarios, and provenance schema

**Files:**
- Create: `src/local_llm_bench/schema.py`
- Create: `src/local_llm_bench/catalog.py`
- Create: `src/local_llm_bench/provenance.py`
- Create: `tests/__init__.py`
- Create: `tests/test_catalog.py`
- Create: `tests/test_provenance.py`

**Interfaces:**
- `load_catalog(root: Path) -> Catalog` loads `profiles/models.toml`, `profiles/engines.toml`, and `profiles/runs.toml`, validates every reference, and reports missing fields with profile IDs.
- `load_scenario(path: Path) -> Scenario` loads either natural multi-turn prompts or a frozen-history replay fixture.
- `build_manifest(profile: RunProfile, scenario: Scenario, *, repo_root: Path, run_id: str, started_at: str) -> dict` records resolved settings and environment metadata without response content or secret values.
- `redact(value: object, *, home: Path, secret_values: set[str]) -> object` recursively removes credentials and replaces local home paths with `$HOME`.
- `ModelProfile` fields are `id`, `repo_id`, `model_path_env`, `revision`, `quantization`, `drafter_repo_id`, `drafter_path_env`, `drafter_revision`, `drafter_bits`, and `drafter_block_size`.
- `EngineProfile` fields are `id`, `kind`, `version`, optional `requirements_file`, `api_base`, and optional `api_key_env`.
- `RunProfile` fields are `id`, `model_profile_id`, `engine_profile_id`, `scenario_id`, `context`, `server_max_tokens`, `prompt_cache_gib`, `mlx_cache_gib`, `parallel`, optional `memory_limit_gb`, `draft_mode` (`draft`, `serial`, or `native_mtp`), optional sampling/thinking/output overrides, `seed`, and `repeat`.
- `Scenario` fields are `id`, `mode`, `system`, `turns`, `temperature`, `thinking`, `thinking_budget`, `request_max_tokens`, and optional `expected_fields`.
- Effective per-request values use run-profile overrides first, then scenario defaults. Engine launch defaults apply only when the run profile does not specify a server setting.

- [ ] **Step 1: Write failing catalog tests** for a valid multi-engine/multi-model profile, missing model reference, invalid context or memory limit, unsupported draft mode, and replay fixture with a missing turn.
- [ ] **Step 2: Run `python3 -m unittest tests.test_catalog -v` and verify each failure is caused by missing catalog/schema behavior.**
- [ ] **Step 3: Implement TOML/JSON schema loading and validation in `schema.py` and `catalog.py` using `tomllib`; do not add a runtime parsing dependency.**
- [ ] **Step 4: Write failing provenance tests** for commit/dirty-state capture, unknown model revision, config/tokenizer fingerprint metadata, secret redaction, and `$HOME` path redaction.
- [ ] **Step 5: Run `python3 -m unittest tests.test_provenance -v` and verify the expected failures.**
- [ ] **Step 6: Implement `build_manifest` and `redact`; full weight-file hashing remains opt-in.**
- [ ] **Step 7: Run `python3 -m unittest tests.test_catalog tests.test_provenance -v`, then commit this task.**

### Task 2: Streaming client, run records, and reports

**Files:**
- Create: `src/local_llm_bench/sse.py`
- Create: `src/local_llm_bench/client.py`
- Create: `src/local_llm_bench/runner.py`
- Create: `src/local_llm_bench/report.py`
- Create: `tests/test_sse.py`
- Create: `tests/test_runner.py`

**Interfaces:**
- `parse_sse(lines: Iterable[bytes], started_at: float, *, clock: Callable[[], float]) -> RunResult` requires a complete `[DONE]` and finish reason; it stores response text only in memory unless explicitly requested.
- `RunResult` includes transient `response_text`, content SHA-256, finish reason, token usage, cached tokens, client TTFT, first visible content, client decode span/rate, total request time, and optional server/draft metrics. The report writer strips `response_text` unless `save_responses=True`.
- Client decode rate is `max(completion_tokens - 1, 0) / (last_nonempty_generation_time - first_nonempty_generation_time)` when both timestamps and usage exist; otherwise null.
- `BenchmarkRunner.run(profile: RunProfile, scenario: Scenario, *, output_root: Path, repeat: int, replay_from: Path | None = None, save_responses: bool = False) -> Path` creates a unique run directory with manifest, `turns.csv`, `results.json`, and `report.md`.

- [ ] **Step 1: Write failing SSE tests** for multiple content chunks, reasoning-only first chunk, final usage, missing usage, missing finish reason, error event, and truncated stream.
- [ ] **Step 2: Run `python3 -m unittest tests.test_sse -v` and confirm failures describe the absent parser/metric behavior.**
- [ ] **Step 3: Implement `RunResult` parsing in `sse.py` and request streaming in `client.py`; preserve server metrics separately from client metrics.**
- [ ] **Step 4: Write failing runner tests** with a local stub HTTP server for natural multi-turn, frozen-history replay, repeated runs, null metrics, report file creation, and optional structured `expected_fields` checks kept separate from speed metrics.
- [ ] **Step 5: Run `python3 -m unittest tests.test_runner -v` and confirm the runner failures are expected.**
- [ ] **Step 6: Implement the runner and report writer; exclude response text and reasoning from default persisted output.**
- [ ] **Step 7: Run the full unit suite, then commit this task.**

### Task 3: One-command setup/run and engine adapters

**Files:**
- Create: `bench.sh`
- Create: `src/local_llm_bench/cli.py`
- Create: `src/local_llm_bench/adapters/base.py`
- Create: `src/local_llm_bench/adapters/tensorfold.py`
- Create: `src/local_llm_bench/adapters/omlx.py`
- Create: `src/local_llm_bench/bootstrap.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_tensorfold_adapter.py`
- Create: `tests/test_omlx_adapter.py`
- Create: `tests/test_end_to_end.py`

**Interfaces:**
- `EngineAdapter` exposes `prepare(profile)`, `start(profile) -> RunningEngine`, `wait_ready(running, timeout)`, `verify_drafter(running, profile) -> DraftStatus`, and `stop(running)`.
- Adapters build argv arrays and environment maps; no profile value is interpolated into a shell command.
- `bench.sh run --profile ID --scenario ID [--repeat N] [--download-models] [--save-responses]` bootstraps the selected pinned environment, validates model files, starts the adapter, calls Task 2's runner, and stops the owned process in `finally`.
- `bench.sh list` prints available profiles; `bench.sh doctor --profile ID` performs preflight without loading model weights.
- The TensorFold adapter launches the profile's pinned CLI with its context, cache, parallelism, memory ceiling, and draft arguments. The oMLX adapter uses an isolated benchmark base path and never writes to or stops the daily `~/.omlx` service.

- [ ] **Step 1: Write failing CLI and adapter tests** using fake executables and an in-process stub API: setup-once behavior, profile resolution, readiness timeout, stop-on-error, no-shell argv construction, missing model preflight, draft fallback rejection, and one full run that writes then sanitizes a result and stops the owned service.
- [ ] **Step 2: Run `python3 -m unittest tests.test_cli tests.test_tensorfold_adapter tests.test_omlx_adapter -v` and confirm expected failures.**
- [ ] **Step 3: Implement the adapter contract and bootstrapper with isolated per-profile virtual environments and pinned requirement files.**
- [ ] **Step 4: Implement CLI `list`, `doctor`, and `run`; always stop only the process instance started by this run.**
- [ ] **Step 5: Implement TensorFold and oMLX adapters, including oMLX path-ancestor guard, generated API key, log-based drafter confirmation, and clean shutdown.**
- [ ] **Step 6: Run adapter tests and the full unit suite, then commit this task.**

### Task 4: Initial profiles, scenarios, results, and publication docs

**Files:**
- Create: `profiles/models.toml`
- Create: `profiles/engines.toml`
- Create: `profiles/runs.toml`
- Create: `scenarios/chat.json`
- Create: `scenarios/coding.json`
- Create: `scenarios/replay/qwen38-coding.json`
- Create: `requirements/tensorfold-v0341.txt`
- Create: `requirements/tensorfold-v0361.txt`
- Create: `requirements/tensorfold-v0363.txt`
- Create: `requirements/omlx-v070rc1.json`
- Create: `README.md`
- Create: `docs/methodology.md`
- Create: `docs/results/initial-results.md`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `NOTICE.md`
- Test: `tests/test_distributed_profiles.py`

- [ ] **Step 1: Write failing profile integration tests** proving the Qwen/TensorFold versions, Qwen/oMLX 28GB and 32GB cases, and Bonsai packed/widened cases resolve without hard-coded model paths; assert all scenarios share the same schema.
- [ ] **Step 2: Run `python3 -m unittest tests.test_distributed_profiles -v` and verify expected missing profile/data failures.**
- [ ] **Step 3: Add initial profiles and prompts from the existing benchmark evidence; use sanitized fixed histories and keep personal paths/secrets out.**
- [ ] **Step 4: Add pinned dependency snapshots, Chinese-first README, methodology and result summaries with exact sample counts and limitations.**
- [ ] **Step 5: Run the full unit suite, shell syntax checks, JSON/TOML validation, CLI dry-run for every published run profile, and secret/path scan; commit this task.**

### Task 5: Whole-project verification and final review

**Files:**
- Verify: all files and tests in the repository.
- Create: no new behavior unless review finds a defect.

- [ ] **Step 1: Run `python3 -m unittest discover -s tests -v`.**
- [ ] **Step 2: Run `./bench.sh list` and `./bench.sh doctor --profile` for each runnable profile; do not run expensive model downloads.**
- [ ] **Step 3: Run `python3 -m unittest tests.test_end_to_end -v`; confirm the stub engine creates a manifest, CSV, JSON, and Markdown report, redacts secrets, and stops the owned service.**
- [ ] **Step 4: Review Git status and generated files; verify no weights, `.omlx-benchmark`, secrets, raw private DM transcript, or user-specific absolute paths are tracked.**
- [ ] **Step 5: Run whole-branch code review and address load-bearing findings; do not create or push a GitHub remote in this task.**
