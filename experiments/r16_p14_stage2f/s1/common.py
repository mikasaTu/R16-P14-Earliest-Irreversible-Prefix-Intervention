"""S1 durable I/O and outcome-blind execution boundaries."""
from __future__ import annotations
import hashlib, json, multiprocessing as mp, os, queue, sys, traceback, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[3]
for stage in ("a","b","c","d"):
    sys.path.insert(0,str(ROOT/"experiments"/f"r16_p14_stage2{stage}"))
TASKS=("put_the_cream_cheese_in_the_bowl","put_the_bowl_on_the_plate")
SEEDS=(7,17,29)
OPERATORS=("fresh_h4","fresh_h16","hold_1+fresh_h4","rollback_1+fresh_h16")
def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()
def atomic_json(path,value,immutable=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if path.exists() and immutable:
        if path.read_text()!=data:raise RuntimeError(f"immutable artifact differs: {path}")
        return
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    with tmp.open("w") as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def split_for(i):
    if not 0<=i<100:raise ValueError(i)
    return "infrastructure" if i<10 else "calibration" if i<50 else "evaluation" if i<90 else "reserve"
def guard_execution(now=None):
    now=now or datetime.now(ZoneInfo("Asia/Shanghai"))
    minute=now.hour*60+now.minute
    # Stop admitting branches five minutes before each user blackout.
    if 565<=minute<580 or 1165<=minute<1180:
        raise RuntimeError("BLACKOUT: refuse new work 09:25-09:40 / 19:25-19:40 Beijing")
    heartbeat=os.environ.get("S1_CONTROL_HEARTBEAT")
    if heartbeat:
        from datetime import timezone
        p=Path(heartbeat)
        if not p.exists():raise RuntimeError("external controller heartbeat missing")
        h=json.loads(p.read_text());age=(datetime.now(timezone.utc)-datetime.fromisoformat(h["time"])).total_seconds()
        if age>90:raise RuntimeError("external controller heartbeat stale")
    marker=os.environ.get("S1_STOP_FILE")
    if marker and Path(marker).exists():raise RuntimeError("external S1 stop marker: "+Path(marker).read_text()[:200])
def _spawn_entry(q,module,function,kwargs):
    try:
        import importlib
        from .assets import configure_assets
        configure_assets()
        out=getattr(importlib.import_module(module),function)(**kwargs)
        q.put({"ok":True,"result":out})
    except BaseException:
        q.put({"ok":False,"error":traceback.format_exc()})
def spawn_call(module,function,kwargs,timeout=1800,cancel_event=None):
    guard_execution()
    ctx=mp.get_context("spawn");q=ctx.Queue()
    p=ctx.Process(target=_spawn_entry,args=(q,module,function,kwargs))
    p.start()
    deadline=time.monotonic()+timeout
    try:
        while True:
            guard_execution()
            if cancel_event is not None and cancel_event.is_set():raise RuntimeError("cancelled active child")
            if time.monotonic()>deadline:raise TimeoutError(f"spawn exceeded {timeout}s")
            try:result=q.get(timeout=1);break
            except queue.Empty:
                if not p.is_alive():raise RuntimeError(f"child exited before result: {p.exitcode}")
    except BaseException:
        if p.is_alive():p.terminate()
        p.join(10)
        if p.is_alive():p.kill();p.join(10)
        raise
    p.join(30)
    if p.is_alive():p.terminate();p.join();raise RuntimeError("child did not exit")
    if not result["ok"]:raise RuntimeError(result["error"])
    if p.exitcode!=0:raise RuntimeError(f"child exit {p.exitcode}")
    return result["result"]
