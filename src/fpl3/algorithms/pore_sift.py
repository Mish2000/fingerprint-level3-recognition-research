"""Pore SIFT driver (project environment side).

Checks the pinned upstream revisions, then runs `pore_sift_worker.py` in the
separate environment, one subprocess per chunk of work (G3, E5).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

WORKER = Path(__file__).with_name("pore_sift_worker.py")


def resolve(config: dict, repo_root: Path) -> dict:
    return {
        "python": config["python"],
        "survey_dir": str((repo_root / config["survey_dir"]).resolve()),
        "dahia_dir": str((repo_root / config["dahia_dir"]).resolve()),
        "model_sha256": config["model_sha256"],
        "survey_revision": config["survey_revision"],
        "dahia_revision": config["dahia_revision"],
    }


def check_upstream(settings: dict) -> None:
    for key in ("survey", "dahia"):
        head = subprocess.run(
            ["git", "-C", settings[f"{key}_dir"], "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if head != settings[f"{key}_revision"]:
            raise RuntimeError(f"{key} checkout is at {head}, expected {settings[f'{key}_revision']}")


def runtime_versions(settings: dict) -> dict:
    code = (
        "import json, sys, numpy, cv2, torch, torchvision;"
        "print(json.dumps({'python': sys.version.split()[0], 'numpy': numpy.__version__, 'opencv': cv2.__version__,"
        " 'torch': torch.__version__, 'torchvision': torchvision.__version__}))"
    )
    return json.loads(subprocess.run([settings["python"], "-c", code], capture_output=True, text=True, check=True).stdout)


def run_parallel(
    command: str, items: list[dict], workers: int, settings: dict, features_dir: Path, work_dir: Path
) -> tuple[list[dict], float]:
    """Split `items` (images for 'extract', pairs for 'compare') into contiguous chunks, one worker each.
    Returns the results in input order and the wall-clock seconds."""
    work_dir.mkdir(parents=True, exist_ok=True)
    size = -(-len(items) // workers)
    chunks = [items[i : i + size] for i in range(0, len(items), size)]
    processes = []
    start = time.perf_counter()
    for index, chunk in enumerate(chunks):
        job = {
            "survey_dir": settings["survey_dir"],
            "dahia_dir": settings["dahia_dir"],
            "model_sha256": settings["model_sha256"],
            "features_dir": str(features_dir),
            "threads": 1,
            "out_path": str(work_dir / f"{command}-{index:03d}.json"),
            ("images" if command == "extract" else "pairs"): chunk,
        }
        job_path = work_dir / f"{command}-{index:03d}.job.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        log = open(work_dir / f"{command}-{index:03d}.log", "w", encoding="utf-8")
        processes.append((subprocess.Popen([settings["python"], str(WORKER), command, "--job", str(job_path)],
                                           stdout=log, stderr=subprocess.STDOUT), log, job["out_path"]))
    results = []
    for process, log, out_path in processes:
        code = process.wait()
        log.close()
        if code != 0:
            raise RuntimeError(f"pore SIFT worker failed ({command}); see {log.name}")
        results.extend(json.loads(Path(out_path).read_text(encoding="utf-8")))
    return results, time.perf_counter() - start
