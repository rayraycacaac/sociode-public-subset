# SociODE

Code accompanying **SociODE: Continuous-Time Social Propagation Cascade Generation via Adaptive Multi-Agent Dynamics**.

This release includes the graph ODE, shared response prototypes, behavioral history pooling, and cumulative-intensity sampler. A small example covers data preparation, temporal training, and cascade generation.

Some components are shared with concurrent internal projects and other ongoing work, which limits their release. We provide the subset that can be distributed independently, together with an executable example. The full text-encoding pipeline, LLM routing and distillation, cross-event history maintenance, trained research checkpoints, and named baseline adapters are outside this release. The example uses synthetic data and does not reproduce the paper's Weibo results.

## Installation

Use Python 3.10–3.12 and install the pinned dependency:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, replace the activation command with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The GPU environment for the research experiments is an NVIDIA GeForce RTX 4090. The public example runs on CPU and was tested with Python 3.12.7 and PyTorch 2.6.0. It requires no language-model download or API access.

## Run the example

```bash
python run.py --config configs/demo.json --output outputs/demo
```

The script generates 12 synthetic cascades on a fixed graph of 10 users and splits them chronologically into 8 training, 2 validation, and 2 test cascades. It then trains the ODE component and a homogeneous exposure-gated Poisson reference, and generates suffixes for the test prefixes. The Poisson model provides a simple reference under the same exposure and sampling interface; it is separate from the named baselines in the paper.

For this example, behavioral histories are empty and source embeddings are zero. A deterministic policy selects an eligible parent and supplies demonstration text. Training covers the temporal likelihood, using RK4 quadrature for the compensator. These settings are recorded in `configs/demo.json`.

Results are written to `outputs/demo/`:

- `data/`: processed splits, graph, and preprocessing counts.
- `<seed>/<method>/`: checkpoint, training history, validation loss, generated traces, and Size/Depth/IAT JSD diagnostics.
- `summary.json` and `environment.json`: configuration, per-seed results, and software versions.

The configuration separates data, training, and rollout seeds. Rollouts share a checkpoint within each training seed. Training uses a fixed epoch count; validation losses are recorded after fitting. IAT bins are fixed in `sociode/metrics.py`. The included [example output](examples/smoke-summary.json) was produced by this command.

Run the numerical and preprocessing checks with:

```bash
python -m unittest discover -s tests -v
```

## Code structure

| File | Purpose |
| --- | --- |
| `sociode/data.py` | Cascade validation, first-participation filtering, root connectivity, and chronological splits |
| `sociode/core.py` | History pooling, state and prototype initialization, graph dynamics, and residual updates |
| `sociode/training.py` | Temporal likelihood and training loop |
| `sociode/sampler.py` | Exposure-gated cumulative-intensity sampling |
| `sociode/baseline.py` | Homogeneous Poisson reference |
| `sociode/metrics.py` | Distribution diagnostics |

Edges point from a posting user to a user who can be exposed to that post. Event files use elapsed seconds; the solver uses hours. Each user participates at most once per cascade. Exponential thresholds, accumulated intensity, and continuous states are retained across participation events.

See [method notes](docs/METHOD.md) for the correspondence with the manuscript and the implementation choices used here.

## Using other data

```bash
python run.py --input cascades.jsonl --graph graph.json --config configs/demo.json --output outputs/custom
```

The [data documentation](docs/DATA.md) describes the input format and research-data access. Training requires verified observation coverage through the configured censoring horizon. The time of the last recorded event alone is insufficient. Supply a graph with documented provenance and source timestamps that permit unambiguous chronological splits.

The synthetic example is distributed with the code. Original Weibo records and follower lists are not included; access and redistribution remain subject to the original providers' terms. Further details are in the [reproduction notes](docs/REPRODUCING.md).

## License

The released code and synthetic example are provided under the [MIT license](LICENSE). External datasets and pretrained models retain their respective licenses.
