"""Reconstruction selection and a non-controlling historical stopping observer."""
from __future__ import annotations
import json
import os
import math
from dataclasses import asdict
from sae_pipeline.sae.train import ObjectiveEarlyStop
from sae_pipeline.sae.checkpoint import write_raw_export


def observe_selection(site, evaluation, rule):
    state = site.selection_state
    objective = evaluation.mixed_objective
    if not math.isfinite(objective):
        raise ValueError('Cannot select a nonfinite reconstruction')
    if state.get('latest', {}).get('step') == site.step:
        return
    payload = dict(step=site.step, tokens=site.tokens_seen(), objective=objective,
                   metrics={lang: {**asdict(metric), 'validation_objective': evaluation.objective[lang]}
                            for lang, metric in evaluation.metrics.items()})
    directory = site.out_dir / 'selection'
    directory.mkdir(exist_ok=True)
    if state.get('best') is None or objective < state['best']['objective']:
        relative = f'selection/best_step_{site.step:07d}.safetensors'
        destination = site.out_dir / relative
        temporary = destination.with_suffix('.partial.safetensors')
        write_raw_export(site.sae, temporary, input_scale=site.normalization_scale)
        os.replace(temporary, destination)
        payload['export'] = relative
        state['best'] = dict(payload)
    observer = ObjectiveEarlyStop()
    observer.load_state_dict(state.get('observer', {}))
    # The historical trainer did not feed its initial baseline into plateau state.
    if site.step>0 and site.step % rule.evaluation_interval_steps==0:
        _, reduced, stopped = observer.update(objective, rule, eligible=site.step >= rule.minimum_steps)
    else:
        reduced = stopped = False
    state['observer'] = observer.state_dict()
    if stopped and state.get('first_stop') is None:
        relative = f'selection/first_stop_step_{site.step:07d}.safetensors'
        destination = site.out_dir / relative
        temporary = destination.with_suffix('.partial.safetensors')
        write_raw_export(site.sae, temporary, input_scale=site.normalization_scale)
        os.replace(temporary, destination)
        state['first_stop'] = {**payload, 'export': relative,
                               'virtual_lr_scale': observer.lr_scale,
                               'best_available': dict(state['best']),
                               'trajectory': 'fixed_lr_observation_not_adaptive_lr_replay'}
    state['latest'] = {**payload, 'virtual_lr_scale': observer.lr_scale,
                       'virtual_lr_reduced': reduced, 'virtual_stop': stopped}
    temporary = directory / 'selection.json.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, directory / 'selection.json')


def prune_selection(site):
    """Only after the training cursor and selection state are committed together."""
    state = site.selection_state
    protected = set()
    if state.get('best'):
        protected.add(state['best']['export'])
    if state.get('first_stop'):
        protected.add(state['first_stop']['best_available']['export'])
    for path in (site.out_dir / 'selection').glob('best_step_*.safetensors'):
        if str(path.relative_to(site.out_dir)).replace('\\', '/') not in protected:
            path.unlink()
