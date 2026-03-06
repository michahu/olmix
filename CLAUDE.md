# Olmix — Project Guide

Toolkit for optimizing pretraining data mixtures. Learns from small-scale proxy experiments ("swarms") to predict how data mixing ratios affect downstream performance, then proposes optimized mixtures for full-scale training.

## Development

```bash
uv pip install -e ".[dev]"    # install with dev deps
make run-checks               # mirrors CI: format-check → lint → typecheck → test
```

CI runs on Python 3.11. Workflow: `.github/workflows/main.yml`.

### Make targets

| Target | What it does |
|---|---|
| `make run-checks` | All CI checks in order (use before pushing) |
| `make format` | Auto-fix formatting + lint |
| `make typecheck` | `pyright olmix` |
| `make test` | `pytest` |
| `make test-fast` | `pytest -m "not slow"` |

### Pre-commit vs CI

Pre-commit hooks auto-fix formatting but do **not** run pyright. Passing pre-commit does not guarantee CI will pass — always run `make run-checks`.

## Code Conventions

### Circular import guard in `fit/__init__.py`

`fit/__init__.py` must **only** import from `fit.config`. It must **not** eagerly import from `fit.core`, `fit.utils`, `fit.law`, or `fit.loaders` — doing so causes a circular import via `aliases.py`.

### Pydantic config patterns

- Config models use Pydantic v2 (`BaseModel`)
- `FitConfig` in `fit/config.py` is loaded via `FitConfig.from_yaml()`
- `ExperimentConfig` in `aliases.py` decomposes into `InfraConfig`, `TrainingConfig`, `DataConfig`, `SwarmConfig`
- Eval config uses a discriminated union: `EvalConfig = InLoopEvalConfig | OfflineEvalConfig`

## End-to-End Workflow

```
olmix priors compute → olmix generate → olmix launch run → olmix fit
```

1. **Priors** — compute token-count priors for generation config
2. **Generate** — create swarm variant configs from a base mix
3. **Launch** — run proxy training jobs (requires Beaker)
4. **Fit** — regress on swarm results, propose optimized mix weights

Shortcut: if you already have CSV results, skip to `olmix fit` directly.

## Architecture

See [docs/DECISIONS.md](docs/DECISIONS.md) for architectural decisions and rejected approaches.
