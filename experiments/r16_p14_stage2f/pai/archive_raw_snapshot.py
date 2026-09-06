"""Copy persisted Stage-2F raw shards and their exact compressed traces.
Copies a finite snapshot; never changes source files or overwrites different bytes.
Phase2 reads require the same committed diagnostic admission as the runner.
"""
import argparse, concurrent.futures, hashlib, json, os
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def copy_exact(source,target,expected=None):
    if source.stat().st_uid!=2254:raise RuntimeError(f"unexpected source owner: {source}")
    raw=source.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    if expected is not None and digest!=expected:raise RuntimeError(f"source hash mismatch: {source}")
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(target)!=digest:raise RuntimeError(f"refuse changed destination: {target}")
        copied=False
    else:
        temporary=target.with_name(target.name+f".archive-{os.getpid()}.tmp")
        temporary.write_bytes(raw)
        if sha(temporary)!=digest:raise RuntimeError("copy verification failed")
        # Exclusive install ensures another publisher cannot be overwritten.
        try:os.link(temporary,target);copied=True
        except FileExistsError:
            if sha(target)!=digest:raise RuntimeError(f"destination raced: {target}")
            copied=False
        finally:temporary.unlink()
    return dict(path=str(target),sha256=digest,bytes=len(raw),copied=copied)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--target-root",type=Path,required=True)
    p.add_argument("--phase",choices=("phase1","phase2"),required=True)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    if not 1<=a.workers<=16:raise ValueError("workers must be 1..16")
    src=a.source_root.resolve();dst=a.target_root.resolve()
    if src==dst:raise ValueError("source and destination must differ")
    if a.phase=="phase2":
        from s1.matrix import selected_diagnostic_budget
        selected_diagnostic_budget(src)
    paths=sorted((src/a.phase/"shards").glob("*/*.json"))
    def one(path):
        row=json.loads(path.read_text())
        if row.get("status")!="COMPLETE" and row.get("error_type")!="PrefixOutsideTaskHorizon":
            raise RuntimeError(f"unknown failure cannot be archived as completed: {path}")
        files=[]
        if row.get("status")=="COMPLETE":
            trace=Path(row["trace_path"]).resolve()
            rel=trace.relative_to(src)
            if rel.parts[0] not in ("phase1","phase2") or rel.parts[1]!="contact_topology":
                raise RuntimeError("trace outside exact phase topology scope")
            files.append(copy_exact(trace,dst/rel,row["trace_sha256"]))
        files.append(copy_exact(path,dst/path.relative_to(src)))
        return files
    files=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,group in enumerate(pool.map(one,paths),1):
            files.extend(group)
            if i%500==0:print(json.dumps(dict(rows=i,total=len(paths))),flush=True)
    unique={f["path"]:f for f in files}
    receipt=dict(status="PASS",scope="finite_persisted_snapshot_not_experiment_completion",
      phase=a.phase,rows=len(paths),files=len(unique),bytes=sum(x["bytes"] for x in unique.values()),
      source_root=str(src),target_root=str(dst),manifest=sorted(unique.values(),key=lambda x:x["path"]))
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="manifest"}),flush=True)
if __name__=="__main__":main()
