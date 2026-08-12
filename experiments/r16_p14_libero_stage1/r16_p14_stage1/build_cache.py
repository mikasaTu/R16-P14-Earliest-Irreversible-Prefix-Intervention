from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_feature_cache, cache_path_for
from .settings import DEFAULT_CACHE_ROOT, DEFAULT_LIBERO_CONFIG, DEVELOPMENT_TASKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(DEVELOPMENT_TASKS))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--libero-config", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument("--demo-count", type=int, default=50)
    parser.add_argument("--max-states-per-demo", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for task_name in args.tasks:
        target = cache_path_for(task_name, args.cache_root)
        if target.is_file():
            print(f"CACHE_ALREADY_COMPLETE task={task_name} path={target}", flush=True)
            continue
        print(f"BUILD_CACHE task={task_name} path={target}", flush=True)
        build_feature_cache(
            task_name,
            output_path=target,
            demo_count=args.demo_count,
            max_states_per_demo=args.max_states_per_demo,
            config_dir=args.libero_config,
        )
        print(f"CACHE_COMPLETE task={task_name} path={target}", flush=True)


if __name__ == "__main__":
    main()
