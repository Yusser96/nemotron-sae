"""Register extraction's durable files in place; never rename an open mmap."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from sae_pipeline.extraction.rolling import atomic_json, file_lock


def tensor_keys(settings):
    return {name:f"L{spec['layer']}_{spec['kind']}" for name,spec in settings['sites'].items()}


def entry_for(path, settings, ids, start, count, *, source_digest=None, keys=None, extractor_validated=False):
    """Register a shard; only a successful maintained-extractor receipt may skip its duplicate finite scan.

    Recovery and external files keep full validation by default. Hashes, token
    identity and shape checks remain independent of finite-value validation.
    """
    path=Path(path); keys=tensor_keys(settings) if keys is None else keys
    def digest(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
    hashes={}
    with safe_open(str(path),framework='pt',device='cpu') as handle:
        saved=handle.get_tensor('token_ids')
        if not torch.equal(saved.long(),ids.long()):raise RuntimeError('Archived training token IDs differ')
        token_hash=digest(saved)
        for name,key in keys.items():
            value=handle.get_tensor(key)
            if value.shape!=(ids.numel(),settings['sites'][name]['width']):raise RuntimeError('Activation shape differs: '+name)
            if not extractor_validated:
                for chunk in value.split(8192):
                    if not torch.isfinite(chunk).all():raise FloatingPointError('Nonfinite activation: '+name)
            hashes[name]=digest(value)
    return dict(start_frame=start,frame_count=count,path=str(path),bytes=path.stat().st_size,
                site_sha256=hashes,token_ids_sha256=token_hash,
                source_digest=source_digest or settings['source_digest'],tensor_keys=keys,
                site_representations={name:settings['sites'][name]['kind'] for name in keys})


def bind_publications(pool, reservations, output, settings):
    with file_lock(pool.root/'pool.lock'):
        state=pool.read()
        for index,(key,*_) in enumerate(reservations):
            state['reservations'][key].update(path=str(Path(output)/(f'{index:06d}' if settings.get('publication_layout')=='per_site' else f'{index:06d}.safetensors')),
                 tensor_keys=tensor_keys(settings),source_digest=settings['source_digest'])
        pool._save(state)


def recover_publications(pool,settings,archive):
    pending=sorted(pool.read()['reservations'].items(),key=lambda item:item[1]['start_frame'])
    gap=False
    for key,reservation in pending:
        if 'path' not in reservation:raise RuntimeError('Unbound publication reservation needs explicit recovery mapping')
        path=Path(reservation['path'])
        if path.with_suffix(path.suffix+'.partial').exists():raise RuntimeError('Incomplete publication requires inspection: '+str(path))
        if not path.exists():gap=True;continue
        if gap:raise RuntimeError('Publication exists after a missing earlier range')
        start=reservation['start_frame'];count=reservation['frame_count']
        ids=torch.cat([archive.frame(index) for index in range(start,start+count)],dim=0)
        if path.is_dir():
            receipt=json.loads((path/'receipt.json').read_text())
            receipt.update(path=str(path),bytes=sum(p.stat().st_size for p in path.iterdir() if p.is_file()))
            subset={**settings,'sites':{name:settings['sites'][name] for name in reservation['tensor_keys']}}
            entry=entry_from_receipt(receipt,subset,ids,start,count)
        else:
            entry=entry_for(path,settings,ids,start,count,source_digest=reservation['source_digest'],keys=reservation['tensor_keys'])
        pool.publish(key,entry)
    # Existing durable outputs have been published; only empty tail reservations remain.
    pool.recover_reservations()


def entry_from_receipt(receipt, settings, ids, start, count):
    """Register trusted extractor output using writer hashes, without activation readback."""
    if receipt.get('version') != 2 or receipt.get('finite_validation') != 'device':
        raise ValueError('Per-site registration requires device-validated v2 output')
    keys=tensor_keys(settings);files=receipt['files'];path=Path(receipt['path'])
    if set(files) != {'token_ids',*keys.values()}:
        raise ValueError('Publication site set differs from requested hooks')
    if files['token_ids']['shape'] != list(ids.shape):
        raise ValueError('Publication token geometry differs')
    expected=hashlib.sha256(memoryview(ids.contiguous().view(torch.uint8).numpy())).hexdigest()
    if files['token_ids']['sha256'] != expected:
        raise ValueError('Publication token IDs differ from archived tokens')
    for name,key in keys.items():
        if files[key]['shape'] != [ids.numel(),settings['sites'][name]['width']]:
            raise ValueError('Publication width differs: '+name)
    for key,item in files.items():
        file=Path(item['path'])
        if file.parent.resolve()!=path.resolve() or file.name!=key+'.safetensors' or file.is_symlink():
            raise ValueError('Publication file outside its immutable directory')
        if file.stat().st_size!=item['bytes']:
            raise ValueError('Publication file size differs')
    return dict(format='per_site_v2',start_frame=start,frame_count=count,path=str(path),bytes=receipt['bytes'],
        site_paths={name:files[key]['path'] for name,key in keys.items()},token_ids_path=files['token_ids']['path'],
        site_sha256={name:files[key]['sha256'] for name,key in keys.items()},token_ids_sha256=expected,
        tensor_keys=keys,finite_validation='device',source_digest=settings['source_digest'],
        site_representations={name:settings['sites'][name]['kind'] for name in keys})
