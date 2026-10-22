import hashlib
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from sae_pipeline.extraction.rolling import RollingPool
from sae_pipeline.extraction.rolling_reader import RollingCacheStream


def entry(folder,start,count=1):
    path=folder/f'{start}.safetensors';ids=torch.arange(start*8,(start+count)*8).reshape(-1,2).long()
    rows=torch.arange(start*8,(start+count)*8).float().reshape(-1,1)
    save_file({'token_ids':ids,'S':rows},str(path))
    sha=lambda x:hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()
    return dict(start_frame=start,frame_count=count,path=str(path),bytes=path.stat().st_size,site_sha256={'S':sha(rows)},token_ids_sha256=sha(ids))


def setup(tmp_path,consumers=('a','b')):
    cache=tmp_path/'cache';cache.mkdir();e=entry(cache,0)
    pool=RollingPool.create(tmp_path/'pool',cap_bytes=2*e['bytes'],owned_roots=[cache],
          consumers={c:dict(checkpoint_dir=tmp_path/c) for c in consumers},initial_entries=[e],frame_tokens=8)
    return pool,cache,e


def commit(pool,tmp_path,consumer,frame):
    folder=tmp_path/consumer;folder.mkdir(exist_ok=True)
    name=f'sae_step_{frame*2:07d}.safetensors';(folder/name).write_bytes(b'weights')
    torch.save({},folder/name.replace('sae_step_','trainer_state_step_').replace('.safetensors','.pt'))
    path=folder/'cursor.pt'
    torch.save(dict(token_stream=dict(pool_id=pool.read()['pool_id'],frame=frame),tokens_processed=frame*8,checkpoints={'S':name}),path)
    pool.commit(consumer,path)


def test_backpressure_waits_for_both_durable_consumers(tmp_path):
    pool,cache,e=setup(tmp_path)
    e1=entry(cache,1);reservation=pool.reserve(1,1,e1['bytes']);pool.publish(reservation,e1)
    assert pool.reserve(2,1,e1['bytes']) is None
    commit(pool,tmp_path,'a',1)
    assert Path(e['path']).exists() and pool.reserve(2,1,e1['bytes']) is None
    commit(pool,tmp_path,'b',1)
    assert not Path(e['path']).exists()
    assert pool.reserve(2,1,e1['bytes']) is not None
    assert pool.used_bytes(pool.read())<=pool.read()['cap_bytes']


def test_missing_fill_and_missing_pair_are_fatal(tmp_path):
    pool,cache,e=setup(tmp_path)
    p=tmp_path/'bad.pt';torch.save(dict(token_stream=dict(pool_id=pool.read()['pool_id'],frame=1),tokens_processed=8,checkpoints={'S':'sae_step_0000002.safetensors'}),p)
    with pytest.raises(FileNotFoundError):pool.commit('a',p)
    Path(e['path']).unlink()
    with pytest.raises(FileNotFoundError):pool.lookup(0)


def test_reader_refresh_exact_resume_and_pinned_prefix(tmp_path):
    pool,cache,e=setup(tmp_path,consumers=('a',))
    stream=RollingCacheStream(pool.root,['S'],'cpu',idle_seconds=0)
    ids=stream.next_batch(4);values=stream.extract(ids).activations['S'].clone();cursor=stream.state_dict()
    assert torch.equal(values[:,0],torch.arange(8).float())
    with pytest.raises(TimeoutError):stream.next_batch(4)
    e1=entry(cache,1);reservation=pool.reserve(1,1,e1['bytes']);pool.publish(reservation,e1)
    second=RollingCacheStream(pool.root,['S'],'cpu',idle_seconds=0);second.load_state_dict(cursor)
    ids_a=stream.next_batch(4);ids_b=second.next_batch(4)
    assert torch.equal(ids_a,ids_b)
    assert torch.equal(stream.extract(ids_a).activations['S'],second.extract(ids_b).activations['S'])
    stream.close();second.close()


def test_bad_publication_does_not_advance_or_drop_reserved_bytes(tmp_path):
    pool,cache,e=setup(tmp_path)
    r=pool.reserve(1,1,e['bytes'])
    e1=entry(cache,1)
    with pytest.raises(RuntimeError):pool.publish(r,{**e1,'start_frame':2})
    assert pool.read()['next_frame']==1 and r in pool.read()['reservations']


def test_duplicate_publication_fails_without_changing_pool(tmp_path):
    pool,cache,e=setup(tmp_path)
    following=entry(cache,1)
    key=pool.reserve(1,1,following['bytes'])
    pool.publish(key,following)
    before=pool.read()
    with pytest.raises(RuntimeError,match='reservation is missing'):
        pool.publish(key,following)
    assert pool.read()==before


def test_atomic_json_concurrent_writers_use_independent_temporaries(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import json
    import threading
    import sae_pipeline.extraction.rolling as rolling
    path=tmp_path/'receipt.json'
    barrier=threading.Barrier(2)
    replace=rolling.os.replace
    sources=[]
    def simultaneous(source,destination):
        if source not in sources:
            sources.append(source)
            barrier.wait(timeout=10)
        replace(source,destination)
    monkeypatch.setattr(rolling.os,'replace',simultaneous)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(rolling.atomic_json,path,{'writer':index}) for index in (1,2)]
        for future in futures:future.result(timeout=15)
    assert len(set(sources))==2
    assert json.loads(path.read_text())['writer'] in (1,2)
    assert not list(tmp_path.glob('*.new'))


def test_simultaneous_publications_and_durable_commits_keep_reservations(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import time
    from sae_pipeline.extraction.rolling import file_lock
    pool,cache,first=setup(tmp_path)
    with file_lock(pool.root/'pool.lock'):
        state=pool.read();state['cap_bytes']=1_000_000;pool._save(state)
    def publisher():
        for frame in range(1,21):
            following=entry(cache,frame)
            key=pool.reserve(frame,1,following['bytes'])
            time.sleep(0.002)
            pool.publish(key,following)
    def consumer(name):
        frame=0
        while frame<21:
            target=pool.read()['next_frame']
            if target>frame:
                commit(pool,tmp_path,name,target)
                frame=target
            else:time.sleep(0.002)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures=[executor.submit(publisher),executor.submit(consumer,'a'),executor.submit(consumer,'b')]
        for future in futures:future.result(timeout=30)
    final=pool.read()
    assert final['next_frame']==final['reserved_frontier']==21
    assert final['reservations']=={}
    assert all(c['frame']==21 for c in final['consumers'].values())


def test_live_tensor_alias_is_read_without_reprojecting(tmp_path):
    cache=tmp_path/'cache';cache.mkdir()
    ids=torch.arange(8).reshape(4,2).long();rows=torch.arange(16).reshape(8,2).float()
    path=cache/'live.safetensors';save_file({'token_ids':ids,'L12_attn_out':rows},str(path))
    sha=lambda x:hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()
    e=dict(start_frame=0,frame_count=1,path=str(path),bytes=path.stat().st_size,
           site_sha256={'legacy':sha(rows)},token_ids_sha256=sha(ids),
           tensor_keys={'legacy':'L12_attn_out'},site_representations={'legacy':'attn_out'})
    pool=RollingPool.create(tmp_path/'pool',cap_bytes=e['bytes'],owned_roots=[cache],
                           consumers={'legacy':dict(checkpoint_dir=tmp_path/'ckpt')},initial_entries=[e],frame_tokens=8)
    stream=RollingCacheStream(pool.root,['legacy'],'cpu')
    batch=stream.next_batch(4)
    assert torch.equal(stream.extract(batch).activations['legacy'],rows)
    stream.close()


def test_completed_publication_recovers_in_place_and_missing_tail_is_released(tmp_path):
    import runpy
    from types import SimpleNamespace
    publication=runpy.run_path(str(Path(__file__).parents[1]/'internal_docs/studies/warm14_20261021/publication.py'))
    pool,cache,first=setup(tmp_path,consumers=('a',))
    with (pool.root/'pool.json').open() as f:
        import json
        state=json.load(f)
    state['cap_bytes']=10000
    from sae_pipeline.extraction.rolling import atomic_json
    atomic_json(pool.path,state)
    key=pool.reserve(1,1,2000);tail=pool.reserve(2,1,2000)
    output=cache/'wave-000000001';output.mkdir()
    settings=dict(sites={'S':dict(layer=2,kind='resid_post',width=1)},source_digest='frozen')
    publication['bind_publications'](pool,[(key,1,1,None),(tail,2,1,None)],output,settings)
    ids=torch.arange(8,16).reshape(4,2).long();rows=torch.arange(8,16).float().reshape(8,1)
    path=output/'000000.safetensors';save_file({'token_ids':ids,'L2_resid_post':rows},str(path))
    with pytest.raises(RuntimeError,match='Recover the existing durable publication'):
        pool.recover_reservations()
    publication['recover_publications'](pool,settings,SimpleNamespace(frame=lambda i:ids))
    assert path.exists() and pool.lookup(1)['path']==str(path)
    assert pool.read()['next_frame']==2 and pool.read()['reserved_frontier']==2
    assert pool.read()['reservations']=={}


def test_legacy_attention_fill_is_projected_to_released_sae_site(tmp_path):
    cache=tmp_path/'legacy';cache.mkdir()
    ids=torch.arange(8).reshape(4,2).long()
    rows=torch.arange(32,dtype=torch.bfloat16).reshape(8,4)
    path=cache/'old.safetensors';save_file({'token_ids':ids,'attn':rows},str(path))
    sha=lambda x:hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest()
    e=dict(start_frame=0,frame_count=1,path=str(path),bytes=path.stat().st_size,
           site_sha256={'attn':sha(rows)},token_ids_sha256=sha(ids),
           site_representations={'attn':'attn_out_prelinear'})
    pool=RollingPool.create(tmp_path/'pool',cap_bytes=e['bytes'],owned_roots=[cache],
                            consumers={'attn':dict(checkpoint_dir=tmp_path/'ckpt')},
                            initial_entries=[e],frame_tokens=8)
    weight=torch.tensor([[1,0,0,0],[0,1,0,0]],dtype=torch.bfloat16)
    stream=RollingCacheStream(pool.root,['attn'],'cpu',attention_projections={'attn':weight})
    batch_ids=stream.next_batch(4)
    actual=stream.extract(batch_ids).activations['attn']
    expected=torch.nn.functional.linear(rows,weight)
    assert actual.shape==(8,2)
    assert torch.equal(actual,expected)
    stream.close()


def test_real_training_resume_matches_uninterrupted_weights(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import runpy
    from sae_pipeline.config import StreamCfg,EarlyStoppingCfg
    from sae_pipeline.sae.train_streaming import run_streaming_training
    from sae_pipeline.sae.checkpoint import load_trainer_state,resolve_checkpoint
    h=SimpleNamespace(**runpy.run_path(str(Path(__file__).with_name('test_train_streaming.py'))))
    import sae_pipeline.sae.train_streaming as training
    monkeypatch.setattr(training,'refresh_plot',lambda *a,**k:None)
    results=[]
    for label,interrupt in [('continuous',False),('resumed',True)]:
        root=tmp_path/label;root.mkdir();cache=root/'cache';cache.mkdir()
        entries=[]
        for i in range(4):
            ids=torch.arange(i*8,(i+1)*8).reshape(4,2).long()
            rows=torch.arange(i*48,(i+1)*48).reshape(8,6).float()/100
            path=cache/f'{i}.safetensors';save_file({'token_ids':ids,'site':rows},str(path))
            sha=lambda t:hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
            entries.append(dict(start_frame=i,frame_count=1,path=str(path),bytes=path.stat().st_size,
                          site_sha256={'site':sha(rows)},token_ids_sha256=sha(ids)))
        pool=RollingPool.create(root/'pool',cap_bytes=sum(e['bytes'] for e in entries),owned_roots=[cache],
                  consumers={'site':dict(checkpoint_dir=root/'streaming')},initial_entries=entries,frame_tokens=8)
        cfg=h._sae_cfg(h._warm_start(root/'warm.safetensors'),batch_size=4,ckpt_every=2,log_every=1)
        val=h._validation_dirs(root)
        site=h._site(root,cfg,val,probe_tokens=8)
        stream=RollingCacheStream(pool.root,['site'],'cpu',idle_seconds=0)
        if interrupt:
            original=stream.next_batch
            def bounded(count):
                if stream.frame>=2:raise TimeoutError('planned interruption')
                return original(count)
            stream.next_batch=bounded
        rule=EarlyStoppingCfg(enabled=True,mode='objective',score='full_training_loss',patience=10,min_lr_scale=1,
                              divergence_l0_factor=None,divergence_objective_factor=None)
        kwargs=dict(stream_cfg=StreamCfg(enabled=True,token_cap=32),early_stopping=rule,tokens_per_fwd=8,seq_len=2,
                    device='cpu',out_root=root/'streaming',name='site')
        done=run_streaming_training(sites={'site':site},token_stream=stream,extractor=stream,**kwargs)
        if interrupt:
            assert not done
            site.close();stream.close()
            state=load_trainer_state(root/'streaming/stream_state_site.pt')
            resume=resolve_checkpoint(root/'streaming',filename=state['checkpoints']['site'],require_trainer_state=True)
            site=h._site(root,cfg,val,probe_tokens=8,resume_checkpoint=resume)
            stream=RollingCacheStream(pool.root,['site'],'cpu',idle_seconds=0)
            assert run_streaming_training(sites={'site':site},token_stream=stream,extractor=stream,**kwargs)
        results.append({k:v.clone() for k,v in site.sae.state_dict().items()})
        site.close();stream.close()
    for key in results[0]:assert torch.equal(results[0][key],results[1][key])
