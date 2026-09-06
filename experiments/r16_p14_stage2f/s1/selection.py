"""Verify a receipt's Git object membership without a checkout inside PAI."""
from __future__ import annotations
import base64,hashlib,json,re
from pathlib import Path
PATH="artifacts/stage2f/phase1/selection_receipt.json"
def oid(kind,data):
    return hashlib.sha1(kind.encode()+b" "+str(len(data)).encode()+b"\0"+data).hexdigest()
def verify_commit_proof(receipt,authorization):
    raw=Path(receipt).read_bytes();a=authorization
    if hashlib.sha256(raw).hexdigest()!=a.get("selection_receipt_sha256"):raise RuntimeError("receipt SHA256 mismatch")
    commit=a.get("git_commit","")
    if not re.fullmatch("[0-9a-f]{40}",commit):raise RuntimeError("invalid selection commit")
    obj=base64.b64decode(a.get("commit_object_base64",""),validate=True)
    if oid("commit",obj)!=commit:raise RuntimeError("selection commit object does not hash to commit")
    root=obj.split(b"\n",1)[0]
    if not root.startswith(b"tree "):raise RuntimeError("commit lacks tree")
    expected=root[5:].decode();proof=a.get("tree_path_proof",[])
    if len(proof)!=len(PATH.split("/")):raise RuntimeError("incomplete receipt tree proof")
    for name,part in zip(PATH.split("/"),proof):
        data=base64.b64decode(part["object_base64"],validate=True)
        if oid("tree",data)!=expected:raise RuntimeError("tree proof hash mismatch")
        cursor=0;found=None
        while cursor<len(data):
            space=data.index(b" ",cursor);nul=data.index(b"\0",space)
            entry_name=data[space+1:nul].decode();child=data[nul+1:nul+21].hex()
            if entry_name==name:found=child
            cursor=nul+21
        if found is None:raise RuntimeError("receipt path absent from committed tree")
        expected=found
    if oid("blob",raw)!=expected:raise RuntimeError("receipt bytes are not the committed blob")
    return True
