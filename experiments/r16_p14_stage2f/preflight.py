#!/usr/bin/env python3
"""Zero-rollout AST preflight; never imports simulator or actor modules."""
from pathlib import Path
import ast, hashlib, json
ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "spawn": "experiments/r16_p14_stage2d/r16_p14_stage2d/fresh_process.py",
    "isolation": "experiments/r16_p14_stage2d/r16_p14_stage2d/isolation.py",
    "d_runtime": "experiments/r16_p14_stage2d/r16_p14_stage2d/runtime.py",
    "c_runtime": "experiments/r16_p14_stage2c/r16_p14_stage2c/runtime.py",
}
def inspect(root=ROOT):
    trees = {k: ast.parse((root / p).read_text()) for k,p in FILES.items()}
    def function(k,name):
        return next(n for n in trees[k].body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name == name)
    def calls(k,name,target):
        return [n.lineno for n in ast.walk(function(k,name)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id == target]
    child_lines=calls("spawn","_child","execute_branch")
    d_inject=calls("d_runtime","execute_branch","apply_perturbation")
    c_inject=calls("c_runtime","reconstruct_to_prefix","apply_perturbation")
    spawn=function("spawn","run_spawned_branch")
    params=[a.arg for a in spawn.args.args + spawn.args.kwonlyargs]
    # The frozen dispatcher has no callable/backend injection surface.
    bound_import=any(isinstance(n,ast.ImportFrom) and n.module == "runtime" and any(a.name == "execute_branch" for a in n.names) for n in trees["spawn"].body)
    # Source evidence is checked structurally across all top-level try bodies.
    direct_d_injection=any(isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id == "apply_perturbation" for t in function("d_runtime","execute_branch").body if isinstance(t,ast.Try) for n in t.body)
    blocked=bool(bound_import and child_lines and d_inject and direct_d_injection)
    return {
        "status": "BLOCKED_BY_FROZEN_SPAWN_ZERO_INJECTION_CONTRACT" if blocked else "REQUIRES_MANUAL_REVIEW",
        "evidence_level": "static_source_audit_only", "rollouts_executed": 0,
        "forbidden_function_executed": False, "stage2d_historical_isolation_pass_unchanged": True,
        "source_files": {k:{"path":p,"sha256":hashlib.sha256((root/p).read_bytes()).hexdigest()} for k,p in FILES.items()},
        "call_chain": {"frozen_imports_execute_branch":bound_import,"spawn_child_execute_branch_lines":child_lines,"stage2d_apply_perturbation_lines":d_inject,"stage2d_injection_is_unconditional_within_try":direct_d_injection,"stage2c_apply_perturbation_lines":c_inject,"spawn_keyword_parameters":params},
        "reason": "Frozen spawn child always invokes Stage-2D execute_branch, which unconditionally invokes apply_perturbation. No supported zero-injection backend argument exists. Zero magnitude still executes forbidden injection code; zero clearance also relocates the blocker.",
        "constraint_source": "User plan background requires direct unchanged Stage-2D fresh_process/isolation reuse; hard constraint 3 says stop/report on implicit injection and do not bypass.",
        "not_a_scientific_K_gate": True,
        "allowed_independent_work": ["phase0a_cpu_reanalysis", "static_preregistration", "report_and_github_publication"],
        "change_needed_before_gpu": "Explicit plan amendment allowing a Stage-2F spawn dispatcher with a zero-injection branch backend while preserving frozen Stage-2D files and isolation requirements.",
    }
def main():
    result=inspect(); out=ROOT/'artifacts/stage2f/preflight/zero_injection_audit.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"call_chain":result["call_chain"]},ensure_ascii=False))
if __name__ == "__main__": main()
