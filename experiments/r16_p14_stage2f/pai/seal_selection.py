"""Seal a published calibration receipt; run only after its Git commit."""
from pathlib import Path
import argparse,base64,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[3]
REL="artifacts/stage2f/phase1/selection_receipt.json"
def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT)
def seal(commit,receipt):
    receipt=Path(receipt);raw=receipt.read_bytes()
    if git("show",commit+":"+REL)!=raw:raise RuntimeError("receipt bytes differ from commit")
    git("fetch","origin","main")
    subprocess.run(["git","merge-base","--is-ancestor",commit,"origin/main"],cwd=ROOT,check=True)
    obj=git("cat-file","commit",commit);tree=obj.split(b"\n",1)[0][5:].decode();proof=[]
    for name in REL.split("/"):
        data=git("cat-file","tree",tree);proof.append({"object_base64":base64.b64encode(data).decode()})
        cursor=0;found=None
        while cursor<len(data):
            sp=data.index(b" ",cursor);nul=data.index(b"\0",sp);entry=data[sp+1:nul].decode()
            if entry==name:found=data[nul+1:nul+21].hex()
            cursor=nul+21
        if found is None:raise RuntimeError("receipt absent from tree")
        tree=found
    result={"git_commit":commit,"selection_receipt_sha256":hashlib.sha256(raw).hexdigest(),
      "commit_object_base64":base64.b64encode(obj).decode(),"tree_path_proof":proof,
      "verified_origin_main":git("rev-parse","origin/main").decode().strip()}
    out=receipt.with_name("selection_authorization.json")
    text=json.dumps(result,indent=2)+"\n"
    if out.exists() and out.read_text()!=text:raise RuntimeError("selection authorization immutable")
    out.write_text(text);print(out)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--commit",required=True);p.add_argument("--receipt",required=True);a=p.parse_args();seal(a.commit,a.receipt)
