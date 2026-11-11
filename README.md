# SociODE: public component release

This repository accompanies **SociODE: Continuous-Time Social Propagation Cascade Generation via Adaptive Multi-Agent Dynamics**. It provides a compact implementation of the continuous-time propagation components and a self-contained CPU example covering data preparation, temporal training, cascade generation, and descriptive evaluation.

## Availability and scope

Parts of the implementation are shared with concurrent internal projects and other ongoing work. These dependencies constrain what we can distribute at present. We have therefore made a best effort to release a self-contained subset that allows the numerical propagation components to be inspected and executed independently.

The release contains the history-pooling module, prototype initialization and intensity functions, a graph-conditioned vector field, exposure and single-participation gates, cumulative-intensity sampling, and the log-weight residual update. The runnable example trains the temporal component on artificial cascades with known observation windows. It uses empty behavioral histories and a zero source vector; a deterministic parent-and-text policy exercises the agent interface. These choices keep the example small and remove model-download and API requirements.

The full text-encoding pipeline, LLM routing and distillation, persistent cross-event history maintenance, original trained checkpoints, and adaptations of the named comparison methods are not included. The example does not reproduce the manuscript's Weibo results or sensitivity tables. Its scores describe only the supplied integration example.

Computational environment. The research experiments were conducted on an NVIDIA GeForce RTX 4090 GPU. The public demonstration included in this repository supports CPU execution and does not require a GPU. Its execution settings differ from those of the full experimental pipeline.

## Quick start

Use Python 3.10–3.12. No GPU, external dataset, API key, or downloaded language model is required.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py --config configs/demo.json --output outputs/demo
python -m unittest discover -s tests -v
```

The default run fits two small models on chronological training cascades: the released ODE component and a homogeneous exposure-gated Poisson reference. The latter checks the common training and sampling interface; it is not an implementation of any named baseline in the paper. Validation losses are recorded after training. Test cascades are used only for generation diagnostics. The epoch count is fixed by the configuration; no test-based selection is performed.

Outputs include processed train/validation/test JSONL files, preprocessing counts, model checkpoints, training histories, validation losses, generated event traces, descriptive Size/Depth/IAT JSD values, the resolved configuration, and software versions. IAT bin edges are fixed in `sociode/metrics.py`. The diagnostic statistics are computed per training seed with rollouts pooled within that seed. No paper table or aggregate significance claim is produced.

## Repository layout

| Path | Contents |
| --- | --- |
| `sociode/data.py` | First-participation and root-connectivity processing; chronological splitting; independent artificial fixture |
| `sociode/core.py` | History pooling, initialization, graph ODE, prototypes, residual update |
| `sociode/training.py` | Differentiable temporal likelihood with RK4 compensator quadrature |
| `sociode/sampler.py` | Exposure-gated sampling with retained thresholds and accumulated hazard |
| `sociode/baseline.py` | Homogeneous exposure-gated Poisson reference using the same interface |
| `sociode/metrics.py` | Small-run distribution diagnostics |
| `configs/demo.json` | All demonstration settings and random seeds |
| `tests/` | Numerical and structural checks |
| `docs/` | Method correspondence, data access, and reproduction notes |

The graph direction is `parent user -> exposed user`. All timestamps in event files are elapsed seconds; the numerical solver uses hours and intensities per hour. Each user participates at most once in a cascade. The sampler draws one exponential threshold per user at the observation cutoff and retains that threshold and the accumulated intensity throughout generation.

## Data and configuration

The default fixture has 10 users and 12 cascades, split into 8 training, 2 validation, and 2 test cascades. It is generated independently of the research data. Its graph is specified before any traces are sampled. No Weibo posts, user identifiers, credentials, or internal files are distributed.

For appropriately licensed data in the documented canonical format:

```bash
python run.py --input cascades.jsonl --graph graph.json --config configs/demo.json --output outputs/custom
```

See [data access and format](docs/DATA.md). The minimal training path requires complete observation through the configured censoring horizon; the last recorded event is not evidence of complete observation. Source-time split boundaries must be unambiguous. The supplied graph must have its provenance documented independently; this tool does not infer a follower graph from evaluation traces.

The configuration records `data_seed`, `training_seeds`, and `rollout_seeds` separately. Additional seeds can be added to those lists. Changing them increases the number of integration runs; it does not turn the demonstration into a reproduction of the paper protocol.

## Relationship to the manuscript

[Method correspondence](docs/METHOD.md) identifies the released equations, implementation choices, and omitted components. [Reproduction notes](docs/REPRODUCING.md) explain the numerical checks and the boundary between executable component verification and empirical reproduction. The code is licensed under MIT; external data and pretrained models retain their own terms.
