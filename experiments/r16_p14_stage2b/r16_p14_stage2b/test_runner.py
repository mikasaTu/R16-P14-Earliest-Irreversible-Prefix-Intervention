from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET

from .io_utils import atomic_write_json, atomic_write_text
from .settings import ARTIFACT_ROOT, EXPERIMENT_ROOT


def main() -> None:
    output = ARTIFACT_ROOT / "tests"
    junit = output / "junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(EXPERIMENT_ROOT / "tests"),
        "-q",
        f"--junitxml={junit}",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    atomic_write_text(output / "pytest.log", result.stdout)
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    assert suite is not None
    total = int(suite.attrib.get("tests", 0))
    failed = int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0))
    skipped = int(suite.attrib.get("skipped", 0))
    summary = {
        "schema_version": 1,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "passed": total - failed - skipped,
        "failed": failed,
        "skipped": skipped,
        "total": total,
        "required_contract_count": 20,
        "implemented_contract_test_count": total,
        "pytest_exit_code": result.returncode,
        "command": command,
    }
    atomic_write_json(output / "summary.json", summary)
    print(result.stdout)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
