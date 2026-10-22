"""Bounded activation publication with immutable ranges and durable-consumer backpressure."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def atomic_json(path, value):
    path=Path(path);temporary=path.with_name(path.name+'.new')
    with temporary.open('w') as f:
        json.dump(value,f);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    if os.name!='nt':
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


@contextmanager
def file_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if os.name=='nt':
            import msvcrt
            if f.tell()==0:f.write(b'0');f.flush()
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:
            if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)


class RollingPool:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'pool.json'

    @classmethod
    def create(cls,root,*,cap_bytes,consumers,owned_roots,initial_entries=(),source_digest='',frame_tokens=8192):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        pool=cls(root)
        with file_lock(root/'pool.lock'):
            if pool.path.exists():raise FileExistsError('Never overwrite an existing rolling pool')
            entries=[dict(e) for e in initial_entries]
            frontier=0
            for e in entries:
                if e['start_frame']!=frontier or e['frame_count']<=0:raise ValueError('Noncontiguous initial activation ranges')
                frontier+=e['frame_count']
                if not Path(e['path']).is_file():raise FileNotFoundError(e['path'])
            if sum(e['bytes'] for e in entries)>cap_bytes:raise ValueError('Initial pool exceeds cap')
            value=dict(version=1,pool_id=uuid.uuid4().hex,cap_bytes=cap_bytes,source_digest=source_digest,frame_tokens=frame_tokens,
                       next_frame=frontier,reserved_frontier=frontier,entries=entries,reservations={},gc_pending=[],
                       consumers={name:dict(frame=info.get('frame',0),checkpoint_dir=str(info['checkpoint_dir']),
                                                active=True,done=False) for name,info in consumers.items()},
                       owned_roots=[str(Path(p).resolve()) for p in owned_roots],producer_state='waiting')
            atomic_json(pool.path,value)
        return pool

    def read(self):return json.loads(self.path.read_text())

    def _save(self,v):atomic_json(self.path,v)

    def _delete(self,v,entry):
        p=Path(entry['path'])
        if p.is_symlink():raise RuntimeError('Refuse to reclaim a symlink')
        resolved=p.resolve()
        if p.suffix!='.safetensors' or not any(Path(r) in resolved.parents for r in v['owned_roots']):
            raise RuntimeError('Refuse reclamation outside approved activation directories')
        p.unlink(missing_ok=True)

    def _gc(self,v):
        active=[c['frame'] for c in v['consumers'].values() if c['active'] and not c['done']]
        watermark=min(active) if active else v['next_frame']
        reclaim=[e for e in v['entries'] if not e.get('pinned') and e['start_frame']+e['frame_count']<=watermark]
        if reclaim:
            v['gc_pending'].extend(reclaim)
            v['entries']=[e for e in v['entries'] if e not in reclaim]
            self._save(v)
        for e in list(v['gc_pending']):
            self._delete(v,e);v['gc_pending'].remove(e);self._save(v)

    @staticmethod
    def used_bytes(v):
        return sum(e['bytes'] for e in v['entries']+v['gc_pending'])+sum(e['bytes'] for e in v['reservations'].values())

    def reserve(self,start_frame,frame_count,byte_bound):
        if min(frame_count,byte_bound)<=0:raise ValueError('Invalid publication reservation')
        with file_lock(self.root/'pool.lock'):
            v=self.read();self._gc(v)
            if start_frame!=v['reserved_frontier']:raise RuntimeError('Missing or repeated extraction range')
            if self.used_bytes(v)+byte_bound>v['cap_bytes']:return None
            key=uuid.uuid4().hex
            v['reservations'][key]=dict(start_frame=start_frame,frame_count=frame_count,bytes=byte_bound)
            v['reserved_frontier']+=frame_count
            self._save(v);return key

    def publish(self,reservation,entry):
        with file_lock(self.root/'pool.lock'):
            v=self.read();r=v['reservations'][reservation]
            if entry['start_frame']!=r['start_frame'] or entry['frame_count']!=r['frame_count'] or entry['start_frame']!=v['next_frame']:
                raise RuntimeError('Out-of-order or mismatched durable publication')
            if entry['bytes']>r['bytes'] or Path(entry['path']).stat().st_size!=entry['bytes']:
                raise RuntimeError('Serialized publication exceeds reserved bytes')
            if not entry.get('site_sha256') or not entry.get('token_ids_sha256'):raise ValueError('Publication requires immutable checksums')
            v['entries'].append(dict(entry));v['next_frame']+=entry['frame_count']
            del v['reservations'][reservation];v['producer_state']='producing';self._save(v)

    def lookup(self,frame):
        v=self.read()
        for e in v['entries']:
            if e['start_frame']<=frame<e['start_frame']+e['frame_count']:
                if not Path(e['path']).is_file():raise FileNotFoundError('Missing committed activation fill: '+e['path'])
                return e
        if frame<v['next_frame']:raise FileNotFoundError('Required activation range was reclaimed; exact replay is required')
        if v['producer_state']=='failed':raise RuntimeError('Activation producer failed; preserve consumers and inspect')
        return None

    def commit(self,consumer,stream_state_path,*,done=False):
        from sae_pipeline.sae.checkpoint import load_trainer_state,companion_state_path
        p=Path(stream_state_path);state=load_trainer_state(p)
        cursor=state['token_stream']
        with file_lock(self.root/'pool.lock'):
            v=self.read();c=v['consumers'][consumer]
            if cursor.get('pool_id')!=v['pool_id']:raise ValueError('Committed cursor belongs to a different pool')
            frame=int(cursor['frame'])
            if frame<c['frame'] or frame>v['next_frame']:raise ValueError('Invalid committed activation cursor')
            if not state['checkpoints'] or frame*v['frame_tokens']!=state['tokens_processed']:
                raise ValueError('Committed cursor and training token counts disagree')
            for name in state['checkpoints'].values():
                if Path(name).name!=name:raise ValueError('Invalid checkpoint basename')
                weights=Path(c['checkpoint_dir'])/name
                if not weights.is_file() or not companion_state_path(weights).is_file():raise FileNotFoundError('Incomplete committed resume pair')
            c.update(frame=frame,done=bool(done),proof_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                     tokens_processed=state['tokens_processed'],at=time.time())
            self._save(v);self._gc(v)

    def set_producer_state(self,state):
        with file_lock(self.root/'pool.lock'):
            v=self.read();v['producer_state']=state;self._save(v)

    def recover_reservations(self):
        """Only while holding the exclusive producer allocation lease."""
        with file_lock(self.root/'pool.lock'):
            v=self.read()
            # Unpublished files remain reserved until their owner removes them.
            for r in v['reservations'].values():
                if r.get('path') and Path(r['path']).exists():
                    raise RuntimeError('Recover the existing durable publication before releasing its reservation')
                path=self.root/'cache'/f"range-{r['start_frame']:09d}.safetensors"
                if path.exists():self._delete(v,dict(path=str(path)))
                path.with_suffix('.partial.safetensors').unlink(missing_ok=True)
            v['reservations']={};v['reserved_frontier']=v['next_frame'];self._save(v)
