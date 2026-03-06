# Architectural Decisions

Record of key design decisions in olmix, distilled from the `plans/` directory (which is no longer tracked in git but preserved locally and in git history).

## Decisions That Stuck

### Direct olmo-core integration — no custom wrappers
*(plans 002, 003)*

We considered a `SupportedModels` enum and custom model config wrappers but decided against them. olmo-core's model factories are sufficient. Less abstraction = less maintenance burden.

### Chinchilla-based scaling — 20 tokens/param
*(plan 004)*

Token budgets are computed as Chinchilla multiples. `TOKENS_PER_PARAM = 20` is the constant in `aliases.py`. Sequence length is olmo-core's concern, not olmix's.

### Config YAML as source of truth
*(plan 006)*

Constraints (domain bounds, sum-to-one, etc.) live in the config YAML via `ConstraintsConfig` in `fit/config.py`, not as CLI flags. This keeps experiments reproducible from a single file.

### Data vs Mix separation
*(plan 023)*

In `LaunchConfig` (`aliases.py`), `data: DataConfig` describes what data sources exist, while `mix: dict[str, MixEntry]` describes how to combine them. This separation lets the same data definition be reused across different mix experiments.

### Sequential explicit workflows
*(plan 021)*

The CLI exposes four sequential commands: `priors → generate → launch → fit`. Each step produces artifacts consumed by the next. No hidden orchestration — users control the pipeline.

### Self-contained generated configs
*(plan 022)*

Output of `olmix generate` is a set of ready-to-launch YAML configs. No back-references to the generation config are needed at launch time.

### Quality buckets with vigintiles
*(plans 012, 015)*

`QualityConfig` in `aliases.py` supports optional quality-based bucketing of data sources. Uses vigintiles (20 quantile bins). Fields are optional for backward compatibility.

### In-loop eval alignment with olmo-core
*(plans 010, 013)*

`InLoopEvalConfig` in `fit/config.py` uses task families that match olmo-core's built-in eval task list, ensuring metric names from WandB logs map cleanly to fit targets.

## Rejected Approaches

### Custom model abstractions (`SupportedModels` enum)
Considered wrapping olmo-core model configs behind our own enum. Rejected — unnecessary indirection, olmo-core factories are stable and sufficient.

### Grouped WandB metrics with objective weights
Considered `GroupedWandbMetrics` and `ObjectiveWeights` classes for weighted multi-objective optimization. Rejected — overengineered. Simpler to specify task families directly in eval config.

### WandB-only fit data source
Early versions only pulled swarm data from WandB. Too limiting — added CSV support so users can bring data from any source.

### Flat-only mix format
Originally mixes were flat `{domain: weight}` dicts. Nested format was needed for hand-written configs where domains group naturally. `flatten_mix()` in `aliases.py` handles both.

### Constraint fields bouncing between ExperimentConfig and CLI
Constraints were initially CLI arguments, then fields on ExperimentConfig. Settled on a dedicated `FitConfig` YAML section (`ConstraintsConfig`) as the single source of truth.

## Originally Planned, Now Completed

These were proposed in plans and have since been implemented:

- **CSV support** *(plan 016)* — `load_from_csv()` in `fit/loaders.py`, used by the fit CLI
- **ExperimentConfig decomposition** *(plan 017)* — split into `InfraConfig`, `TrainingConfig`, `DataConfig`, `SwarmConfig` in `aliases.py`
- **Fit YAML config** *(plan 018)* — `FitConfig.from_yaml()` in `fit/config.py`, loaded in `fit/cli.py`
- **Typed eval config** *(plan 019)* — discriminated union `EvalConfig = InLoopEvalConfig | OfflineEvalConfig` in `fit/config.py`
