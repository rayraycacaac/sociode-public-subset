"""One-command CPU training and generation for the public component subset."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import torch
from sociode.baseline import ExposurePoisson
from sociode.core import SociODECore
from sociode.data import fixture, prepare, write_json
from sociode.metrics import compare
from sociode.sampler import generate
from sociode.training import initial_inputs, loss, train


def demo_agent(uid, time, parents, p, h):
    """Deterministic interface fixture; this policy is neither an LLM nor a student."""
    parent = max(parents, key=lambda e: (e['t_seconds'], str(e['event_id'])))
    residual = torch.linspace(-.2, .2, len(p)).tolist()
    return {'parent_id': parent['event_id'], 'repost_text': 'Demonstration repost.',
            'strategy_residual': residual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).parent/'configs/demo.json'))
    parser.add_argument('--output', default='outputs/demo')
    parser.add_argument('--input', help='Optional canonical cascade JSONL; requires --graph')
    parser.add_argument('--graph', help='JSON object with users, directed edges and provenance')
    args = parser.parse_args()
    if bool(args.input) != bool(args.graph):
        parser.error('--input and --graph must be supplied together')
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if not (0 <= cfg['tau_hours'] < cfg['end_hours']) or cfg['step_hours'] <= 0:
        parser.error('Invalid integration interval')
    if cfg['epochs'] < 1 or not cfg['training_seeds'] or not cfg['rollout_seeds']:
        parser.error('Specify at least one epoch, training seed and rollout seed')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    if args.input:
        raw = [json.loads(line) for line in Path(args.input).read_text(encoding='utf-8').splitlines() if line.strip()]
        graph = json.loads(Path(args.graph).read_text(encoding='utf-8'))
        identity = 'user-supplied data; component example, not full SociODE'
    else:
        raw, graph = fixture(cfg)
        identity = 'independently generated synthetic fixture'
    splits = prepare(raw, graph, cfg, out/'data')
    users = graph['users']
    lookup = {u: i for i, u in enumerate(users)}
    edges = tuple(torch.tensor([lookup[e[k]] for e in graph['edges']], dtype=torch.long) for k in (0, 1))
    # Validate censoring for every split before fitting anything.
    for rows in splits.values():
        for cascade in rows:
            if not cascade.get('coverage_verified') or cascade.get('observed_until_seconds', -1) < cfg['end_hours']*3600:
                raise ValueError('Known observation coverage through end_hours is required; recorded span is insufficient')
    report = {'identity': identity, 'config': cfg, 'runs': []}
    for seed in cfg['training_seeds']:
        torch.manual_seed(seed)
        core = SociODECore(cfg['prototypes'], cfg['state_width'])
        models = {'ode_component': core, 'exposure_poisson_reference': ExposurePoisson()}
        for name, model in models.items():
            history = train(model, core, splits['train'], users, edges, cfg)
            with torch.no_grad():
                validation_nll = sum(float(loss(model, core, c, users, edges, cfg))
                                     for c in splits['validation'])/len(splits['validation'])
            directory = out/str(seed)/name
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = directory/'checkpoint.pt'
            torch.save({'model_state': model.state_dict(), 'initializer_state': core.state_dict(),
                        'config': cfg, 'training_seed': seed, 'identity': identity}, checkpoint)
            results, references, generations = [], [], []
            for cascade in splits['test']:
                prefix = [e for e in cascade['events'] if e['t_seconds'] <= cfg['tau_hours']*3600]
                target = [e for e in cascade['events'] if e['t_seconds'] <= cfg['end_hours']*3600]
                for rollout in cfg['rollout_seeds']:
                    with torch.no_grad():
                        h, p, memory, source = initial_inputs(core, len(users))
                        generated = generate(model, h, p, memory, source, edges, prefix, users,
                                             cfg['tau_hours'], cfg['end_hours'], rollout,
                                             demo_agent, variant='fixed', step_hours=cfg['step_hours'])
                    if not generated['completed']:
                        raise RuntimeError('Sampler stopped before completion')
                    results.append({'cascade_id': cascade['cascade_id'], 'rollout_seed': rollout, **generated})
                    references.append(target)
                    generations.append(prefix + generated['events'])
            write_json(directory/'training.json', {'epochs': history, 'validation_temporal_nll_per_user': validation_nll})
            write_json(directory/'generated.json', results)
            metrics = compare(references, generations)
            write_json(directory/'metrics.json', metrics)
            report['runs'].append({'training_seed': seed, 'method': name, 'metrics': metrics,
                                  'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
            print('seed=%s method=%s training complete; validation temporal NLL/user=%.6f' % (seed, name, validation_nll), flush=True)
    write_json(out/'summary.json', report)
    write_json(out/'environment.json', {'python': platform.python_version(), 'torch': str(torch.__version__),
                                      'device': 'cpu', 'threads': torch.get_num_threads()})
    print('Finished. Outputs: ' + str(out))


if __name__ == '__main__':
    main()
