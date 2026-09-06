#!/usr/bin/env python3
"""Verify Stage-2F's published artifacts, frozen protocol, and execution disposition.

Run in a complete clone; standard library only. This does not run a simulator.
"""
import argparse, hashlib, importlib.util, json, math, subprocess, sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = "508e8a5c88780066a00d994c32d673d558594867"
BLOCK = "BLOCKED_BY_FROZEN_SPAWN_ZERO_INJECTION_CONTRACT"

# The formal and diagnostic selection artifacts intentionally have separate,
# fixed paths. Keep these constants local to the verifier so a diagnostic row
# cannot silently make the formal receipt acceptable.
FORMAL_RECEIPT_REL = "artifacts/stage2f/phase1/selection_receipt.json"
FORMAL_AUTH_REL = "artifacts/stage2f/phase1/selection_authorization.json"
DIAGNOSTIC_RECEIPT_REL = "artifacts/stage2f/phase1/diagnostic_selection_receipt.json"
DIAGNOSTIC_AUTH_REL = "artifacts/stage2f/phase1/diagnostic_selection_authorization.json"
DIAGNOSTIC_RECEIPT_PATH = DIAGNOSTIC_RECEIPT_REL


def read(p): return json.loads((ROOT / p).read_text())
def require(ok,msg):
    if not ok: raise RuntimeError(msg)


def _load_selection_module(root):
    """Load the in-repository proof helper without third-party imports."""
    path = Path(root) / "experiments/r16_p14_stage2f/s1/selection.py"
    spec = importlib.util.spec_from_file_location("stage2f_selection_proof", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("selection proof module missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_matrix_module(root):
    """Import matrix from this checkout so diagnostic admission is source-bound."""
    package_root = str(Path(root) / "experiments/r16_p14_stage2f")
    previous = list(sys.path)
    try:
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        import importlib

        module = importlib.import_module("s1.matrix")
        expected = (Path(root) / "experiments/r16_p14_stage2f/s1/matrix.py").resolve()
        actual = Path(getattr(module, "__file__", "")).resolve()
        require(actual == expected, "diagnostic matrix source does not match verifier root")
        return module
    finally:
        sys.path[:] = previous


def _read_jsonl(path, label):
    """Read a published JSONL file strictly, with line-specific failures."""
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        raise RuntimeError(f"{label} unreadable: {exc}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise RuntimeError(f"{label} contains a blank line at {number}")
        try:
            row = json.loads(line)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} invalid JSON at line {number}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"{label} line {number} is not an object")
        rows.append(row)
    require(rows, f"{label} is empty")
    return rows


def _diagnostic_summary_marked(summary):
    """Recognize only explicit diagnostic Phase-2 summary markers."""
    return (
        summary.get("diagnostic_continuation") is True
        or summary.get("diagnostic_atlas") is True
        or summary.get("diagnostic_only") is True
        or "diagnostic_selection_receipt_sha256" in summary
        or "diagnostic_selection_binding" in summary
    )


def _scan_rows_for_diagnostic(path):
    """Find diagnostic markers without changing the ordinary formal path.

    Formal verification historically did not parse atlas rows. If a formal
    file has no diagnostic marker, malformed content therefore remains outside
    this new branch. A file containing a diagnostic marker is admitted to the
    diagnostic branch and parsed strictly after selection proof.
    """
    path = Path(path)
    if not path.exists():
        return False, None
    try:
        raw = path.read_text()
    except OSError as exc:
        raise RuntimeError(f"cannot inspect {path}: {exc}") from exc
    marker_tokens = (
        '"diagnostic_continuation"',
        '"diagnostic_atlas"',
        '"diagnostic_selection_',
    )
    if not any(token in raw for token in marker_tokens):
        return False, None
    # Defer JSON parsing until the diagnostic selection has been verified.
    # This keeps the publication gate fail-closed before consuming row data.
    return True, None


def _validate_diagnostic_summary(summary):
    require(
        _diagnostic_summary_marked(summary),
        "diagnostic Phase-2 summary marker missing",
    )
    require(
        summary.get("confirmatory") is False,
        "diagnostic Phase-2 summary must set confirmatory=false",
    )


def _verify_formal_selection(root, atlas):
    """Preserve the pre-existing formal selection check byte-for-byte in spirit."""
    receipt = Path(root) / FORMAL_RECEIPT_REL
    auth = Path(root) / FORMAL_AUTH_REL
    if not Path(atlas).exists():
        return
    require(receipt.exists() and auth.exists(), "evaluation lacks committed calibration selection")
    authorization = json.loads(auth.read_text())
    require(
        authorization["selection_receipt_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "selection receipt mismatch",
    )
    require(len(authorization.get("git_commit", "")) == 40, "selection commit missing")
    proof_module = _load_selection_module(root)
    require(proof_module.verify_commit_proof(receipt, authorization), "selection Git object proof invalid")


def _verify_diagnostic_selection(root):
    """Verify independent diagnostic proof and re-run matrix admission."""
    root = Path(root)
    receipt = root / DIAGNOSTIC_RECEIPT_REL
    auth = root / DIAGNOSTIC_AUTH_REL
    require(
        receipt.exists() and auth.exists(),
        "diagnostic Phase-2 evidence lacks independent selection receipt/auth",
    )
    try:
        authorization = json.loads(auth.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("diagnostic authorization unreadable") from exc
    receipt_sha256 = hashlib.sha256(receipt.read_bytes()).hexdigest()
    require(
        authorization.get("selection_receipt_sha256") == receipt_sha256,
        "diagnostic selection receipt SHA256 mismatch",
    )
    require(
        len(authorization.get("git_commit", "")) == 40,
        "diagnostic selection commit missing",
    )
    proof_module = _load_selection_module(root)
    require(
        proof_module.verify_commit_proof(receipt, authorization, diagnostic=True),
        "diagnostic selection Git object proof invalid",
    )
    matrix = _load_matrix_module(root)
    try:
        budget = matrix.selected_diagnostic_budget(root / "artifacts/stage2f")
    except Exception as exc:
        raise RuntimeError(f"diagnostic selection matrix admission failed: {exc}") from exc
    require(isinstance(budget, dict), "diagnostic matrix admission returned no budget")
    for field in ("tail_horizon", "action_budget", "policy_call_cap"):
        require(field in budget, f"diagnostic matrix budget missing {field}")
    return {
        "receipt_sha256": receipt_sha256,
        "receipt": json.loads(receipt.read_text()),
        "budget": dict(budget),
    }


def _validate_diagnostic_rows(rows, selection):
    """Require a receipt binding on every diagnostic published row."""
    receipt = selection["receipt"]
    receipt_sha256 = selection["receipt_sha256"]
    budget = selection["budget"]
    require(rows, "diagnostic atlas has no rows")
    for number, row in enumerate(rows, 1):
        require(
            row.get("diagnostic_continuation") is True,
            f"diagnostic atlas row {number} lacks diagnostic_continuation=true",
        )
        require(
            row.get("diagnostic_selection_receipt_sha256") == receipt_sha256,
            f"diagnostic atlas row {number} receipt SHA256 mismatch",
        )
        if "diagnostic_selection_receipt_path" in row:
            require(
                row.get("diagnostic_selection_receipt_path") == DIAGNOSTIC_RECEIPT_PATH,
                f"diagnostic atlas row {number} receipt path mismatch",
            )
        if "confirmatory" in row:
            require(
                row.get("confirmatory") is False,
                f"diagnostic atlas row {number} is confirmatory",
            )
        binding = row.get("diagnostic_selection_binding")
        if binding is not None:
            require(isinstance(binding, dict), f"diagnostic atlas row {number} binding malformed")
            for field in ("input_sha256", "protocol_sha256", "original_rank"):
                if field in binding:
                    require(
                        binding[field] == receipt.get(field),
                        f"diagnostic atlas row {number} binding mismatch: {field}",
                    )
        configured = row.get("configured_budget")
        if configured is not None:
            require(isinstance(configured, dict), f"diagnostic atlas row {number} configured budget malformed")
            for field in ("tail_horizon", "action_budget", "policy_call_cap"):
                if field in configured:
                    require(
                        configured[field] == budget[field],
                        f"diagnostic atlas row {number} budget mismatch: {field}",
                    )
        for field in ("tail_horizon", "action_budget", "policy_call_cap"):
            if field in row:
                require(
                    row[field] == budget[field],
                    f"diagnostic atlas row {number} budget mismatch: {field}",
                )


def _verify_phase2_artifacts(phase2_summary, atlas, reference, root=ROOT):
    """Validate formal or diagnostic Phase-2 publication metadata.

    This helper performs no simulator work. The diagnostic branch verifies
    the independent receipt and matrix selection before accepting rows.
    """
    atlas = Path(atlas)
    reference = Path(reference)
    summary_marked = _diagnostic_summary_marked(phase2_summary)
    if summary_marked:
        # The summary is sufficient to select the diagnostic path. Do not
        # inspect row contents before the independent selection gate.
        atlas_marked, atlas_rows = False, None
        reference_marked, reference_rows = False, None
    else:
        atlas_marked, atlas_rows = _scan_rows_for_diagnostic(atlas)
        reference_marked, reference_rows = _scan_rows_for_diagnostic(reference)
    diagnostic = summary_marked or atlas_marked or reference_marked
    if not diagnostic:
        _verify_formal_selection(root, atlas)
        return {"diagnostic": False, "receipt_sha256": None, "row_count": None}

    _validate_diagnostic_summary(phase2_summary)
    require(atlas.exists(), "diagnostic Phase-2 summary has no atlas_rows.jsonl")
    # Verify the independent selection before consuming published row data.
    selection = _verify_diagnostic_selection(root)
    # A marker may have been found in the lightweight scan; re-read every
    # published file strictly once the diagnostic branch is selected.
    atlas_rows = _read_jsonl(atlas, str(atlas))
    all_rows = list(atlas_rows)
    if reference.exists():
        all_rows.extend(_read_jsonl(reference, str(reference)))
    summary_receipt = phase2_summary.get("diagnostic_selection_receipt_sha256")
    if summary_receipt is not None:
        require(
            summary_receipt == selection["receipt_sha256"],
            "diagnostic Phase-2 summary receipt SHA256 mismatch",
        )
    selected_budget = phase2_summary.get("selected_budget")
    if selected_budget is not None:
        require(
            selected_budget == selection["budget"],
            "diagnostic Phase-2 summary selected budget mismatch",
        )
    _validate_diagnostic_rows(all_rows, selection)
    return {
        "diagnostic": True,
        "receipt_sha256": selection["receipt_sha256"],
        "row_count": len(all_rows),
    }


def _scientific_success(phase_status, diagnostic):
    """Diagnostic positives remain outside the formal scientific K3 result."""
    if diagnostic:
        return False
    return phase_status.get("K3") == "NONTRIVIAL_OPERATOR_RELATIVITY"


def verify(skip_worktree=False):
    checks=[]
    manifest=ROOT/'artifacts/stage2f/SHA256SUMS'
    declared={}
    for line in manifest.read_text().splitlines():
        digest,rel=line.split("  ",1)
        path=Path(rel)
        require(not path.is_absolute() and ".." not in path.parts,"unsafe checksum path")
        require(rel not in declared,"duplicate checksum entry")
        require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'bad digest')
        require((ROOT/path).is_file(),f'missing {rel}')
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,f'hash mismatch: {rel}')
        declared[rel]=digest
    expected=set()
    for folder in ['artifacts/stage2f','experiments/r16_p14_stage2f']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p != manifest and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo'}:
                expected.add(p.relative_to(ROOT).as_posix())
    expected.add('scripts/verify_r16p14_stage2f.py')
    require(set(declared)==expected,f'checksum coverage mismatch: {sorted(set(declared)^expected)}')
    checks.append({'check':'output_hashes_and_coverage','files':len(declared)})
    forbidden=[f'{base}/stage2{x}' if base=='artifacts' else f'{base}/r16_p14_stage2{x}' for base in ['artifacts','experiments'] for x in 'abcde']
    committed=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',BASE,'HEAD','--',*forbidden],text=True).strip()
    require(not committed,f'frozen committed paths changed: {committed}')
    if not skip_worktree:
        dirty=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only','HEAD','--',*forbidden],text=True).strip()
        require(not dirty,f'frozen worktree paths changed: {dirty}')
    checks.append({'check':'frozen_predecessor_paths','base':BASE,'working_files_checked':not skip_worktree})
    p=ROOT/'experiments/r16_p14_stage2f/preflight.py'
    spec=importlib.util.spec_from_file_location('stage2f_static_preflight',p); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    live=module.inspect(ROOT); stored=read('artifacts/stage2f/preflight/zero_injection_audit.json')
    require(live==stored,'source audit does not reproduce')
    require(live['status']==BLOCK and not live['forbidden_function_executed'],'blocker disposition mismatch')
    checks.append({'check':'static_injection_call_chain','status':BLOCK})
    prereg=read('artifacts/stage2f/preflight/prereg_freeze.json')
    require(hashlib.sha256((ROOT/'experiments/r16_p14_stage2f/PREREG_S1.md').read_bytes()).hexdigest()==prereg['prereg_sha256'],'preregistered K criteria changed')
    amendment=read('artifacts/stage2f/preflight/execution_amendment_receipt.json')
    require(hashlib.sha256((ROOT/'experiments/r16_p14_stage2f/AMENDMENT_S1_EXECUTION.md').read_bytes()).hexdigest()==amendment['amendment_sha256'],'execution amendment changed')
    phase_status={}
    phase_summaries={}
    for phase,gate in [('phase0b','K1'),('phase1','K2'),('phase2','K3')]:
        summary=read(f'artifacts/stage2f/{phase}/summary.json')
        phase_summaries[phase]=summary
        phase_status[gate]=summary.get('gate_status',summary.get('status',summary.get('execution_status')))
    pools=list((ROOT/'artifacts/stage2f/phase0b/init_pool').glob('*.json'))
    require(len(pools)==2,'two fixed init pools required')
    for pool in pools:
        d=json.loads(pool.read_text());states=d['states']
        require(len(states)==100 and len({x['state_hash'] for x in states})==100,'invalid or duplicate reset pool')
        require([x['init_state_id'] for x in states]==list(range(100)),'init ids changed')
        for item in states:
            i=item['init_state_id'];split='infrastructure' if i<10 else 'calibration' if i<50 else 'evaluation' if i<90 else 'reserve'
            require(item['split']==split,'split changed')
    jobs=read('experiments/r16_p14_stage2f/pai/jobs.json')
    require(jobs['max_concurrent_jobs']==2 and jobs['max_gpu_per_job']==2 and jobs['gpu_hours_limit']==20,'resource caps drifted')
    ids=[j['job_id'] for j in jobs['jobs']]
    require(len(ids)==len(set(ids)),'duplicate job identity')
    for job in jobs['jobs']:
        require(int(job.get('gpus',2))<=2,'per-job GPU cap exceeded')
        if job.get('status')=='Succeeded':require(job.get('persisted_completion_verified') is True,'Succeeded without artifact verification')
    atlas=ROOT/'artifacts/stage2f/phase2/atlas_rows.jsonl'
    reference=ROOT/'artifacts/stage2f/phase2/reference_rows.jsonl'
    phase2_result=_verify_phase2_artifacts(
        phase_summaries['phase2'],
        atlas,
        reference,
        ROOT,
    )
    checks.append({'check':'frozen_S1_protocol_and_available_execution_evidence','phase_status':phase_status,'PAI_jobs':len(ids),'phase2_selection_mode':'diagnostic' if phase2_result['diagnostic'] else 'formal_or_none','phase2_published_rows':phase2_result['row_count']})
    summary=read('artifacts/stage2f/phase0a/summary.json')
    require(summary.get('diagnostic_only') is True,'Phase0A not diagnostic-only')
    require((ROOT/'artifacts/stage2f/phase0a/table.csv').stat().st_size>0,'missing Phase0A table')
    require(summary['control_candidates']==['fixed_delay_1','fixed_delay_2','fixed_delay_4','fixed_delay_8','immediate_fresh_h16'],'incorrect control expansion')
    require(summary['bootstrap_replicates']==10000 and summary['bootstrap_seed']==216214,'bootstrap contract drift')
    require(summary['A1_exact_reproduction'] is True,'A1 not reproduced')
    require(summary['G0_1']=='INCONCLUSIVE','unexpected G0 claim')
    require(len(summary['absolute_comparisons_same_support'])==4,'missing or duplicate paired absolute comparisons')
    for item in summary['source_inputs']['files'].values():
        require(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],'Phase0A source input drift')
    require(hashlib.sha256((ROOT/'scripts/run_r16p14_stage2e_s0.py').read_bytes()).hexdigest()==summary['source_script_sha256'],'frozen A source script drift')
    for c in summary['absolute_comparisons_same_support']:
        d=c['paired_cluster_delta']
        require(math.isclose(c['k_cluster_safe_success']-c['comparator_cluster_safe_success'],d['estimate'],abs_tol=1e-12),'absolute/delta estimand mismatch')
        require(d['replicates']==10000 and d['seed']==216214,'comparison bootstrap drift')
        require(d['ci95'][0]<0<d['ci95'][1],'unexpected significance claim')
    checks.append({'check':'phase0a_frozen_inputs_A1_controls_and_estimands','input_files':4,'A1_methods':9,'same_support_comparisons':4})
    checks.append({'check':'phase0a_outputs_present','diagnostic_only':True})
    result = {
        'status':'PASS',
        'scope':'integrity_of_published_evidence_not_a_claim_of_experiment_completion',
        'scientific_S1_success':_scientific_success(
            phase_status,
            phase2_result['diagnostic'],
        ),
        'phase_status':phase_status,
        'checks':checks,
    }
    if phase2_result['diagnostic']:
        result.update(
            diagnostic_continuation=True,
            diagnostic_selection_receipt_sha256=phase2_result['receipt_sha256'],
            scientific_S1_success=False,
        )
    return result
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-worktree',action='store_true',help='Only verify committed frozen-path trees; default also checks tracked working files')
    args=parser.parse_args()
    try: result=verify(args.skip_worktree)
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error':str(exc)},ensure_ascii=False));return 1
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0
if __name__=='__main__': sys.exit(main())
