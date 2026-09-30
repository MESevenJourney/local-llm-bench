# Local LLM benchmark project design

Status: draft for review

## Goal

Create a standalone, GitHub-ready project for reproducible local LLM inference tests. A new reader should be able to install the selected test profile, point it at the required model files or explicitly request their download, run a benchmark, and receive a report that records enough information to repeat the same experiment.

The project must not be tied to Qwen, Bonsai, TensorFold, or oMLX. Those are initial examples. The first supported machine class is Apple Silicon macOS, matching the current M3 Max and M4 Mac mini tests. The core runner should use an OpenAI-compatible local API where possible; small engine adapters will handle server startup, draft controls, readiness checks, and engine-specific measurements.

The current Chinese X Article will describe one Qwen3.8-27B + DFlash2 experiment, show the main workflow, and link to the public benchmark project when it is ready. Bonsai 2 will be a separate model profile and article case.

## Requirements

### Reproducibility

- Keep model profiles separate from engine profiles and test scenarios. A profile describes target model ID, local path or download source, quantization, optional drafter and its settings, context limit, sampling defaults, and output budget.
- A test profile combines a model profile, engine profile, and scenario. Initial examples cover Qwen3.8-27B + DFlash2 on TensorFold and oMLX, plus the available Bonsai 2 TensorFold cases. A profile may mark a combination unsupported or not yet tested.
- Provide one command that prepares the selected software environment and runs a test, for example `./bench.sh run --profile qwen38-tensorfold --scenario coding`. Re-running the same profile should use pinned dependencies and record any user override. If model files are missing, the command gives a profile-specific download option instead of silently fetching large weights.
- Model weights are not stored in the Git repository. The command validates local files; weight download is opt-in because files are large and model licenses vary.
- Offer both natural multi-turn runs, where each engine continues from its own answers, and frozen-history replay, where every engine receives the same messages.
- A profile that requests speculative decoding must verify from engine status or logs that the intended drafter is attached and that drafting occurred. The run must fail or be marked invalid if drafting silently falls back to serial generation.

### Traceability

Each run writes an immutable run directory containing:

- a manifest with project commit, dirty-tree state, scenario/profile IDs and hashes, host chip and memory, macOS and Python versions, engine/runtime/dependency versions, model and drafter repository IDs/revisions, quantization, context and output settings, seed and sampling values, and the exact launch command with secrets removed;
- per-turn metrics in JSON and CSV, plus a Markdown summary;
- sanitized startup and relevant engine logs. The default artifact stores scenario IDs, request parameters, output hashes, and measurements, not response text, hidden reasoning, or API credentials. Retaining response text is an explicit option and does not retain hidden reasoning;
- model config and tokenizer fingerprints, plus file size metadata. Full weight hashing is optional because hashing multi-gigabyte weights adds setup time.

Local absolute paths, API keys, private messages, and model weights must never be included in a shareable result bundle. The repository should ignore runtime environments, local model directories, service settings, and raw run artifacts by default. Shareable bundles include the manifest, per-turn metrics, summary, and sanitized logs; unredacted server logs remain local.

### Metrics and comparison rules

- Record input, cached, and output tokens; client TTFT and first visible content; client decode rate; total request and session time; finish reason; and any server-side metrics supplied by the engine.
- Use one client-side streaming decode formula across engines. Keep server-reported rates in separate fields because engines define them differently. Missing values are null/N/A, never zero.
- When available, record drafted and accepted tokens, memory pressure and prefill throttling events, process-reported memory, and startup calibration. Document what each engine actually reports.
- Do not describe a cross-engine result as a controlled kernel comparison when versions, memory accounting, or runtime stacks differ. Report sample count and run-to-run spread.
- Keep generated-code checks separate from speed metrics. A successful structured field check does not establish that generated code is correct.

### Usability

- Chinese is the primary documentation language for the first release; English can follow.
- The quick start must fit in a small number of commands and clearly distinguish environment setup, model preparation, and test execution.
- A preflight command checks the selected profile, model files, ports, runtime version, memory settings, and required service options before loading a model.
- Engine processes must be started and stopped by their adapter where possible. Existing personal oMLX settings and services must remain untouched; isolated benchmark configuration is required.
- Reports must be readable without Python or shell knowledge. Reproduction instructions should include the exact command and profile used to create each published result.

## Proposed structure

    README.md
    bench.sh
    profiles/models/
    profiles/engines/
    profiles/runs/
    scenarios/
    src/bench/
    adapters/tensorfold/
    adapters/omlx/
    requirements/
    docs/methodology.md
    results/summary/
    .gitignore
    LICENSE

This is a small adapter-based layout, not a general plugin framework. The core runner owns prompts, replay, metrics, manifests, and reports. An engine adapter translates launch, readiness, draft controls, and health checks. Model and scenario additions should normally require profile/data changes rather than Python changes.

Each engine adapter must pass a contract suite against a stub OpenAI-compatible SSE server. Hardware tests are opt-in and run only on a machine supported by that profile. The common tests cover profile validation, metric calculations, replay ordering, manifest completeness, and secret/path redaction.

## Initial workflow

1. Clone the repository and run the selected profile. The CLI creates or verifies a pinned isolated environment and validates model files.
2. If weights are missing, the user may opt into the profile's documented download command. The default run does not download weights silently.
3. The adapter starts the engine with the profile settings, verifies readiness and requested draft mode, then the common runner sends the scenario.
4. The runner saves a manifest, per-request metrics, sanitized logs, and a Markdown summary under a unique run ID.
5. The user can rerun the same command or share a sanitized result directory. The report identifies any profile, dependency, model revision, or hardware difference.

Initial scenarios are the existing eight-turn Chinese chat and synthetic TTL/LRU coding/debugging workload. Engine comparison uses the frozen coding history. Natural multi-turn runs remain available to show practical conversation behavior. Context limits are reported as configured caps; the report must also show the longest input actually tested.

## Public repository and attribution

Do not publish model weights, `.omlx-benchmark`, API keys, the private DM transcript, or raw profiling bundles containing local paths. Keep author screenshots in the X Article only, using the user's original screenshots.

The project may use MIT for newly authored harness code, with clear attribution and retained license notices for copied TensorFold-derived scripts. Model and drafter downloads must link to their source and state that users are responsible for the applicable model terms. Do not claim that the benchmark repository is maintained by TensorFold or oMLX.

GitHub creation and push happen only after the project files and first reproducible example have been reviewed. The X Article should not point to a repository URL until that URL exists.

## Acceptance criteria

- A clean macOS Apple Silicon machine can run the documented setup for one chosen profile without editing source files.
- One CLI invocation can prepare the selected environment, validate prerequisites, start the selected engine, verify the requested draft mode, run a named scenario, stop the owned service, and write a report.
- A second invocation with identical inputs records enough metadata to identify whether the test setup matches the first run.
- The same frozen history can be sent to TensorFold, oMLX, and a future engine adapter without changing the scenario data.
- The Qwen test is represented as an initial profile, not as the repository's only supported model. Bonsai 2 and a future model can be added without changing the report schema.
- Public result bundles contain no private credentials, private conversations, or model weights.
- Documentation distinguishes version comparison, draft-on/serial comparison, and cross-engine comparison, including sample counts and limitations.
- Automated tests cover the common runner and adapter contract without requiring model weights. A hardware smoke test validates each advertised engine/model profile on supported Apple Silicon.

## Review decisions

- The proposed repository name is `local-llm-bench`; it can be renamed before publication.
- The first release targets macOS Apple Silicon and keeps profiles extensible to other local engines.
- The proposed license is MIT for project-authored code, with preserved attribution and license notices for reused code.
