"""Full run of the approved algorithms on both scenarios (B3 full design, decision record 2026-10-07).

    python -m fpl3.experiments.full_run [--workers 12]   start, or continue after an interruption
    python -m fpl3.experiments.full_run --status         progress and the expected finish
    python -m fpl3.experiments.full_run --stop           stop a running run; the start command continues it

Every probe (U, R) is compared with every reference (V) of the same finger position: 582,724
pairs per algorithm. The compact pairs (manifests/pairs.csv) are a subset and run first, so their
results are ready early. Stages: OpenCV SIFT features, OpenCV SIFT compact pairs, Pore SIFT
features, Pore SIFT compact pairs, the other Pore SIFT pairs, the other OpenCV SIFT pairs.

Every stage is cut into chunks and each finished chunk is saved at once, so the start command
continues an interrupted run where it stopped. <out>/progress.html refreshes itself and shows the
expected finish; <out>/progress.json holds the same data. Results go to <out>/report.json and
<out>/scores/<algorithm>.csv; features stay in <out>/features and never enter git (R8).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from multiprocessing import TimeoutError as PoolTimeout
from pathlib import Path

from ..algorithms import opencv_sift, pore_sift
from ..data.build import write_csv
from ..data.pairs import all_impostor_pairs, genuine_pairs
from ..eval.metrics import tar_at_far
from .common import REPO_ROOT, host, image_path, load_toml, read_images, read_pairs, write_json
from .sample_run import read_sample

NAMES = {"opencv-sift": "OpenCV SIFT", "pore-sift": "Pore SIFT"}
FAR_TARGETS = {"compact": (0.01, 0.001), "full": (0.01, 0.001, 0.0001)}  # E2
FAR_RULE = (
    "TAR at FAR x: a pair is accepted when its score is above the lowest threshold at which at most "
    "a share x of the impostor pairs is accepted; a failed comparison is never accepted (E1)."
)
CHUNK_SIZE = {
    ("opencv-sift", "extract"): 25,
    ("opencv-sift", "compare"): 200,
    ("pore-sift", "extract"): 20,
    ("pore-sift", "compare"): 500,
}
# Worker seconds per item while 12 workers run, measured on 2026-10-07 per image set and per probe
# set: Pore SIFT from the timing-pilot logs, OpenCV SIFT from a 3-minute re-measurement. They weigh
# progress and forecast stages that have not started; finished chunks correct them as the run goes.
WORKER_SECONDS = {
    ("opencv-sift", "extract"): {"U": 1.31, "V": 1.31, "R": 0.26},
    ("opencv-sift", "compare"): {"U": 0.687, "R": 0.478},
    ("pore-sift", "extract"): {"U": 15.6, "V": 16.9, "R": 3.2},
    ("pore-sift", "compare"): {"U": 0.381, "R": 0.180},
}
FEATURE_MB = {"opencv-sift": 9.9, "pore-sift": 1.9}  # largest mean feature file per image in the timing pilots
SET_ORDER = {"V": 0, "U": 1, "R": 2}  # dearest first, so cheap items fill the end of a stage
SCORE_FIELDS = ("pair_id", "scenario", "kind", "frgp", "probe_image_id", "reference_image_id", "compact", "status", "reason", "score")
PAGE_EVERY_S = 15
STALE_AFTER_S = 180


def item_id(item: dict) -> str:
    return item["pair_id"] if "pair_id" in item else item["image_id"]


def item_set(item: dict) -> str:
    """Image set of an image, or of a pair's probe."""
    return item["probe_image_id"][0] if "pair_id" in item else item["set"]


@dataclass
class Stage:
    algorithm: str
    kind: str  # "extract" or "compare"
    design: str  # "" for extraction, "compact" or "rest" for comparisons
    items: list[dict]

    @property
    def name(self) -> str:
        return f"{self.algorithm}/{self.kind}" + (f"-{self.design}" if self.design else "")

    @property
    def chunk_size(self) -> int:
        return CHUNK_SIZE[(self.algorithm, self.kind)]

    @property
    def n_chunks(self) -> int:
        return -(-len(self.items) // self.chunk_size)

    def chunks(self) -> list[list[dict]]:
        return [self.items[i : i + self.chunk_size] for i in range(0, len(self.items), self.chunk_size)]

    def expected_seconds(self, items: list[dict]) -> float:
        table = WORKER_SECONDS[(self.algorithm, self.kind)]
        return sum(table[item_set(item)] for item in items)

    def label(self, hebrew: bool = True) -> str:
        part = self.design or self.kind
        if hebrew:
            what = {"extract": "חילוץ מאפיינים", "compact": "השוואות המדגם המצומצם", "rest": "שאר ההשוואות"}[part]
            return f"{NAMES[self.algorithm]} · {what}"
        what = {"extract": "features", "compact": "compact pairs", "rest": "other pairs"}[part]
        return f"{NAMES[self.algorithm]} {what}"


@dataclass
class Context:
    run_dir: Path
    dataset_root: Path
    workers: int
    opencv_seed: int
    pore_settings: dict

    def features_dir(self, algorithm: str) -> Path:
        return self.run_dir / "features" / algorithm


# ---- what to run ----


def with_int_frgp(pairs: list[dict]) -> list[dict]:
    return [{**p, "frgp": int(p["frgp"])} for p in pairs]


def full_design(images: list[dict], protocol: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Images, compact pairs and the other pairs of the full design."""
    full = []
    for name, scenario in sorted(protocol["scenarios"].items()):
        full += genuine_pairs(images, name, scenario["probe"], scenario["reference"])
        full += all_impostor_pairs(images, name, scenario["probe"], scenario["reference"])
    compact = with_int_frgp(read_pairs())
    compact_ids = {p["pair_id"] for p in compact}
    if not compact_ids <= {p["pair_id"] for p in full}:
        raise SystemExit("manifests/pairs.csv has pairs outside the full design")
    used = [r for r in images if r["set"] in ("U", "V", "R") and not int(r["excluded"])]
    return used, compact, [p for p in full if p["pair_id"] not in compact_ids]


def sample_design(name: str) -> tuple[list[dict], list[dict], list[dict]]:
    """A saved sample (manifests/<name>_images.csv and _pairs.csv) as one compare stage per
    algorithm, to check the run end to end on a small scale."""
    rows, pairs = read_sample(name)
    return rows, with_int_frgp(pairs), []


def build_stages(images: list[dict], compact: list[dict], rest: list[dict]) -> list[Stage]:
    def pair_order(p: dict) -> tuple:
        return SET_ORDER[p["probe_image_id"][0]], p["frgp"], p["probe_image_id"], p["reference_image_id"]

    images = sorted(images, key=lambda r: (SET_ORDER[r["set"]], r["image_id"]))
    compact, rest = sorted(compact, key=pair_order), sorted(rest, key=pair_order)
    return [
        Stage("opencv-sift", "extract", "", images),
        Stage("opencv-sift", "compare", "compact", compact),
        Stage("pore-sift", "extract", "", images),
        Stage("pore-sift", "compare", "compact", compact),
        Stage("pore-sift", "compare", "rest", rest),
        Stage("opencv-sift", "compare", "rest", rest),
    ]


def plan_record(stages: list[Stage], ctx: Context, protocol: dict, algorithms: dict) -> dict:
    """Everything that fixes the results; a run directory only continues with the same plan."""
    digest = hashlib.sha256()
    for stage in stages:
        digest.update(f"{stage.name} {stage.chunk_size}\n".encode())
        for item in stage.items:
            digest.update(f"{item_id(item)}\n".encode())
    return {
        "items_sha256": digest.hexdigest(),
        "stages": {s.name: {"items": len(s.items), "chunks": s.n_chunks} for s in stages},
        "dataset_root": ctx.dataset_root.as_posix(),
        "master_seed": protocol["randomness"]["master_seed"],
        "opencv_seed": ctx.opencv_seed,
        "opencv": opencv_sift.cv2.__version__,
        "pore_sift": {k: algorithms["pore_sift"][k] for k in ("survey_revision", "dahia_revision", "model_sha256")},
    }


# ---- chunk store: one JSON file per finished chunk ----


def chunk_file(run_dir: Path, stage: Stage, index: int) -> Path:
    return run_dir / "chunks" / stage.name / f"{index:05d}.json"


def save_chunk(run_dir: Path, stage: Stage, index: int, chunk: list[dict], results: list[dict], started: float, finished: float) -> dict:
    ids = [item_id(item) for item in chunk]
    if [item_id(r) for r in results] != ids:
        raise RuntimeError(f"{stage.name} chunk {index}: the results do not match the chunk")
    meta = {"stage": stage.name, "index": index, "count": len(chunk), "first": ids[0], "last": ids[-1],
            "expected_s": round(stage.expected_seconds(chunk), 3), "started": started, "finished": finished}
    path = chunk_file(run_dir, stage, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({**meta, "results": results}), encoding="utf-8")
    os.replace(temporary, path)
    return meta


def finished_chunks(run_dir: Path, stage: Stage) -> dict[int, dict]:
    folder = run_dir / "chunks" / stage.name
    chunks = stage.chunks()
    done = {}
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        meta = json.loads(path.read_text(encoding="utf-8"))
        del meta["results"]
        index = meta["index"]
        if index >= len(chunks) or (meta["count"], meta["first"], meta["last"]) != (
            len(chunks[index]), item_id(chunks[index][0]), item_id(chunks[index][-1])
        ):
            raise SystemExit(f"{path} does not belong to this run's plan")
        done[index] = meta
    return done


def stage_results(run_dir: Path, stage: Stage) -> dict[str, dict]:
    results = {}
    for path in sorted((run_dir / "chunks" / stage.name).glob("*.json")):
        for result in json.loads(path.read_text(encoding="utf-8"))["results"]:
            results[item_id(result)] = result
    return results


# ---- progress, forecast and the progress page ----


def log(run_dir: Path, message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    for _ in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:  # a reader such as a browser may hold the file for a moment
            time.sleep(0.2)
    os.replace(temporary, path)


class Progress:
    """What is done and what is left; the forecast scales the expected worker seconds of the work
    left by how much longer the finished chunks took than expected."""

    def __init__(self, run_dir: Path, stages: list[Stage], workers: int, started: float):
        self.run_dir, self.stages, self.workers, self.started = run_dir, stages, workers, started
        self.done = {s.name: finished_chunks(run_dir, s) for s in stages}
        self.expected = {s.name: s.expected_seconds(s.items) for s in stages}
        self.state, self.error, self.current = "running", "", None
        self.results: dict = {}
        self.warnings: list[str] = []
        self.last_write = 0.0

    def record(self, stage: Stage, meta: dict) -> None:
        self.done[stage.name][meta["index"]] = meta

    @staticmethod
    def factor(metas: list[dict]) -> float | None:
        expected = sum(m["expected_s"] for m in metas)
        return sum(m["finished"] - m["started"] for m in metas) / expected if expected > 0 else None

    def stage_factor(self, stage: Stage, now: float) -> float:
        """From the stage's own chunks (the last two hours once there are enough), else from the
        stages of the same algorithm and kind, else 1: extraction and comparison keep separate
        paces, so one never forecasts the other."""
        own = list(self.done[stage.name].values())
        recent = [m for m in own if m["finished"] > now - 7200]
        candidates = [
            recent if len(recent) >= self.workers else own,
            [m for s in self.stages if (s.algorithm, s.kind) == (stage.algorithm, stage.kind) for m in self.done[s.name].values()],
        ]
        for metas in candidates:
            factor = self.factor(metas)
            if factor is not None:
                return factor
        return 1.0

    def snapshot(self, now: float) -> dict:
        stages, remaining_total, duration_total = [], 0.0, 0.0
        for s in self.stages:
            metas = list(self.done[s.name].values())
            factor = self.stage_factor(s, now)
            left = s.n_chunks - len(metas)
            expected_left = self.expected[s.name] - sum(m["expected_s"] for m in metas)
            remaining = expected_left * factor / min(self.workers, left) if left else 0.0
            duration = self.expected[s.name] * factor / min(self.workers, max(1, s.n_chunks))
            remaining_total += remaining
            duration_total += max(duration, remaining)
            stages.append({
                "name": s.name,
                "label": s.label(),
                "label_en": s.label(hebrew=False),
                "unit": "images" if s.kind == "extract" else "pairs",
                "items": len(s.items),
                "items_done": sum(m["count"] for m in metas),
                "chunks": s.n_chunks,
                "chunks_done": len(metas),
                "status": "done" if not left else ("running" if s.name == self.current else "waiting"),
                "started": min((m["started"] for m in metas), default=None),
                "finished": max((m["finished"] for m in metas), default=None) if not left else None,
                "remaining_s": remaining,
                "duration_s": duration,
            })
        return {
            "state": self.state,
            "error": self.error,
            "updated": now,
            "started": self.started,
            "workers": self.workers,
            "current": self.current,
            "remaining_s": remaining_total,
            "finish_at": now + remaining_total,
            "percent": 100.0 * (1 - remaining_total / duration_total) if duration_total else 100.0,
            "stages": stages,
            "results": self.results,
            "warnings": self.warnings,
            "commands": {flag: f'"{sys.executable}" -m fpl3.experiments.full_run{flag}' for flag in ("", " --status", " --stop")},
        }

    def write(self, force: bool = False) -> None:
        """Never stops the run: a failed update is logged and the next one tries again."""
        now = time.time()
        if not force and now - self.last_write < PAGE_EVERY_S:
            return
        self.last_write = now
        try:
            snapshot = self.snapshot(now)
            atomic_write(self.run_dir / "progress.json", json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
            atomic_write(self.run_dir / "progress.html", render_page(snapshot))
        except Exception:
            log(self.run_dir, "progress page not updated:\n" + traceback.format_exc())


HEBREW_DAYS = ("שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון")  # datetime.weekday(): Monday = 0
SCENARIOS_HE = {"UxV": "מגולגלת מול מגולגלת", "RxV": "שטוחה מול מגולגלת", "pilot": "מדגם הזמנים"}
UNITS_HE = {"images": "תמונות", "pairs": "זוגות"}
STATUS_HE = {"done": "הסתיים", "running": "בביצוע", "waiting": "ממתין"}


def hebrew_duration(seconds: float) -> str:
    """For example '13 שעות ו-12 דקות', 'שעתיים ודקה', '45 דקות'."""
    hours, minutes = divmod(max(0, round(seconds / 60)), 60)
    hours_text = {0: "", 1: "שעה", 2: "שעתיים"}.get(hours, f"{hours} שעות")
    minutes_text = {0: "", 1: "דקה"}.get(minutes, f"{minutes} דקות")
    if hours_text and minutes_text:
        return f"{hours_text} ו{'-' if minutes_text[0].isdigit() else ''}{minutes_text}"
    return hours_text or minutes_text or "פחות מדקה"


def about(text: str) -> str:
    """'כ-13 שעות', 'כשעתיים'."""
    if text == "פחות מדקה":
        return text
    return ("כ-" if text[0].isdigit() else "כ") + text


def hebrew_time(timestamp: float, now: float) -> str:
    when = datetime.fromtimestamp(timestamp)
    days = (when.date() - datetime.fromtimestamp(now).date()).days
    day = {-1: "אתמול", 0: "היום", 1: "מחר"}.get(days) or f"ביום {HEBREW_DAYS[when.weekday()]} {when.day}.{when.month}"
    return f"{day} בשעה {when:%H:%M}"


PAGE_CSS = """
:root { --bg: #f6f6f4; --card: #ffffff; --text: #1c1c1e; --muted: #66666c; --line: #e2e2de;
  --accent: #2563eb; --accent-soft: #dce6fb; --ok: #15803d; --warn-bg: #fff3dc; --warn: #7a4a00; --bad: #b42318; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #141416; --card: #1d1d20; --text: #ececf0; --muted: #a1a1aa; --line: #303036;
    --accent: #7ba4ff; --accent-soft: #24304a; --ok: #4ade80; --warn-bg: #3a2c10; --warn: #f5c66a; --bad: #ff8a80; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 16px/1.55 "Segoe UI", system-ui, -apple-system, Arial, sans-serif; }
main { max-width: 940px; margin: 0 auto; padding: 28px 16px 48px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.1rem; margin: 0 0 12px; }
h3 { font-size: 1rem; margin: 22px 0 8px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 20px; margin: 16px 0; }
.kicker { margin: 0; color: var(--muted); }
.big { margin: 2px 0 4px; font-size: 2.1rem; font-weight: 700; line-height: 1.25; }
.sub { margin: 0 0 14px; font-size: 1.15rem; }
.bar { height: 12px; background: var(--accent-soft); border-radius: 999px; overflow: hidden; }
.bar > span { display: block; height: 100%; background: var(--accent); }
.bar.small { height: 6px; margin: 4px 0; }
.now { margin: 12px 0 0; }
.muted { color: var(--muted); font-size: .9rem; }
.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.bad { color: var(--bad); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: start; padding: 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: .85rem; color: var(--muted); font-weight: 600; }
td.strong { font-weight: 700; }
tr.done .status { color: var(--ok); }
tr.running .status { color: var(--accent); font-weight: 600; }
tr.waiting { color: var(--muted); }
.banner { background: var(--warn-bg); color: var(--warn); border-radius: 12px; padding: 12px 16px; margin: 16px 0; }
code { direction: ltr; unicode-bidi: isolate; display: inline-block; max-width: 100%; overflow-wrap: anywhere;
  font: .82rem Consolas, "Cascadia Mono", monospace; background: var(--accent-soft); padding: 2px 6px; border-radius: 6px; }
.commands p { margin: 8px 0; }
"""

PAGE_JS = """
(function () {
  var root = document.documentElement;
  var updated = Number(root.dataset.updated) * 1000;
  function tick() {
    var s = Math.max(0, Math.round((Date.now() - updated) / 1000));
    document.getElementById("age").textContent = s < 60 ? "(לפני " + s + " שניות)" : "(לפני " + Math.round(s / 60) + " דקות)";
    document.getElementById("stale").hidden = !(root.dataset.state === "running" && s > STALE);
  }
  tick();
  setInterval(tick, 1000);
})();
"""


def isolate_names(markup: str) -> str:
    """Keep each Latin algorithm name in one piece inside right-to-left text."""
    for name in NAMES.values():
        markup = markup.replace(name, f"<bdi>{name}</bdi>")
    return markup


def results_html(results: dict) -> str:
    blocks = []
    for algorithm, name in NAMES.items():
        entry = results.get(algorithm, {})
        for design, title in (("compact", "המדגם המצומצם"), ("full", "המדגם המלא")):
            if design not in entry:
                continue
            rows, failures = [], 0
            for scenario in sorted(entry[design], key=lambda s: list(SCENARIOS_HE).index(s) if s in SCENARIOS_HE else 99):
                table = entry[design][scenario]
                failures += table["failures"]
                ordered = sorted(table["at_far"].items(), key=lambda item: -float(item[0].rstrip("%")))  # 1 %, 0.1 %, ...
                for i, (far, r) in enumerate(ordered):
                    head = f'<td rowspan="{len(table["at_far"])}">{html.escape(SCENARIOS_HE.get(scenario, scenario))}</td>' if i == 0 else ""
                    rows.append(
                        f'<tr>{head}<td class="num">{far}</td><td class="num strong">{100 * r["tar"]:.1f}%</td>'
                        f'<td class="num">{100 * r["frr"]:.1f}%</td>'
                        f'<td class="num">{r["true_accepts"]:,} מתוך {r["genuine"]:,}</td>'
                        f'<td class="num">{r["false_accepts"]:,} מתוך {r["impostor"]:,}</td></tr>'
                    )
            note = f'<p class="muted">השוואות שנכשלו (נספרות כדחייה): {failures:,}</p>' if failures else ""
            blocks.append(
                f"<h3><bdi>{name}</bdi> · {title}</h3><div class=\"table-wrap\"><table><thead><tr><th>השוואה</th><th>FAR</th>"
                "<th>TAR</th><th>FRR</th><th>זוגות אמת שהתקבלו</th><th>מתחזים שהתקבלו</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>{note}"
            )
    if not blocks:
        return '<p class="muted">התוצאות של המדגם המצומצם יופיעו כאן בסוף שלב ההשוואות שלו.</p>'
    return "".join(blocks) + (
        '<p class="muted">בכל שורה, זוג מתקבל כשהציון שלו גבוה מהסף הנמוך ביותר שבו שיעור המתחזים שמתקבלים '
        "לא עולה על ה-FAR שבשורה. השוואה שנכשלה לא מתקבלת לעולם.</p>"
    )


def render_page(snap: dict) -> str:
    e = html.escape
    now, state, stages = snap["updated"], snap["state"], snap["stages"]
    if state == "running":
        head = (f'<p class="kicker">צפי לסיום</p><p class="big">{e(hebrew_time(snap["finish_at"], now))}</p>'
                f'<p class="sub">נותרו {e(about(hebrew_duration(snap["remaining_s"])))}</p>')
    elif state == "finished":
        end = max((s["finished"] or 0) for s in stages)
        head = f'<p class="kicker">ההרצה הסתיימה</p><p class="big">{e(hebrew_time(end, now))}</p>'
    elif state == "stopped":
        head = '<p class="kicker">ההרצה הופסקה לבקשתך</p><p class="sub">אפשר להמשיך מאותה נקודה עם פקודת ההרצה שבתחתית הדף.</p>'
    else:
        head = (f'<p class="kicker bad">ההרצה נעצרה בגלל שגיאה</p><p class="sub"><bdi>{e(snap["error"])}</bdi></p>'
                '<p class="sub">אחרי הטיפול בשגיאה אפשר להמשיך מאותה נקודה עם פקודת ההרצה שבתחתית הדף.</p>')
    head += f'<div class="bar"><span style="width:{snap["percent"]:.1f}%"></span></div>'
    head += f'<p class="muted">התקדמות כוללת: {snap["percent"]:.0f}% · ההרצה התחילה {e(hebrew_time(snap["started"], now))}</p>'
    current = next((s for s in stages if s["name"] == snap["current"]), None)
    if current and state == "running":
        head += (f'<p class="now">כעת: {isolate_names(e(current["label"]))}, {current["items_done"]:,} מתוך {current["items"]:,} '
                 f'{UNITS_HE[current["unit"]]}</p>')

    rows = []
    for s in stages:
        share = 100 * s["items_done"] / s["items"] if s["items"] else 100
        if s["status"] == "done":
            took = f'ארך {hebrew_duration(s["finished"] - s["started"])}' if s["started"] else ""
        elif s["status"] == "running":
            took = f'נותרו {about(hebrew_duration(s["remaining_s"]))}'
        else:
            took = f'צפוי לקחת {about(hebrew_duration(s["duration_s"]))}'
        rows.append(
            f'<tr class="{s["status"]}"><td>{isolate_names(e(s["label"]))}</td>'
            f'<td><div class="bar small"><span style="width:{share:.1f}%"></span></div>'
            f'<span class="num muted">{s["items_done"]:,} מתוך {s["items"]:,} {UNITS_HE[s["unit"]]}</span></td>'
            f'<td class="status">{STATUS_HE[s["status"]]}</td><td>{e(took)}</td></tr>'
        )
    pairs = sum(s["items"] for s in stages if s["name"].startswith("opencv-sift/compare"))
    commands = snap["commands"]
    updated = datetime.fromtimestamp(now)
    return f"""<!doctype html>
<html lang="he" dir="rtl" data-updated="{now:.0f}" data-state="{e(state)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>ההרצה המלאה</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<main>
<h1>ההרצה המלאה</h1>
<p class="muted"><bdi>OpenCV SIFT</bdi> ו-<bdi>Pore SIFT</bdi> · {pairs:,} זוגות לכל אלגוריתם · {snap["workers"]} תהליכים במקביל</p>
<div id="stale" class="banner" hidden>הדף לא התעדכן כבר יותר מ-{STALE_AFTER_S // 60} דקות. ייתכן שההרצה נעצרה, למשל כי המחשב כבה או נכנס למצב שינה. אפשר להמשיך מאותה נקודה עם פקודת ההרצה שבתחתית הדף.</div>
{"".join(f'<div class="banner">{isolate_names(e(w))}</div>' for w in snap["warnings"])}
<section class="card">{head}</section>
<section class="card"><h2>שלבים</h2><div class="table-wrap"><table>
<thead><tr><th>שלב</th><th>התקדמות</th><th>מצב</th><th>זמן</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p class="muted">הצפי מחושב מהקצב שנמדד עד עכשיו ומתעדכן לאורך ההרצה.</p></section>
<section class="card"><h2>תוצאות</h2>{results_html(snap["results"])}</section>
<section class="card commands"><h2>פקודות</h2>
<p>הרצה, או המשך מאותה נקודה אחרי הפסקה: <code>{e(commands[""])}</code></p>
<p>מצב ההרצה בטרמינל: <code>{e(commands[" --status"])}</code></p>
<p>עצירה: <code>{e(commands[" --stop"])}</code></p>
<p class="muted">את הפקודות מריצים מתיקיית הפרויקט.</p></section>
<p class="muted">עודכן בשעה {updated:%H:%M:%S} <span id="age"></span> · הדף מתרענן מעצמו כל 30 שניות</p>
</main>
<script>{PAGE_JS.replace("STALE", str(STALE_AFTER_S))}</script>
</body>
</html>
"""


def hours_minutes(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


def print_status(run_dir: Path) -> None:
    path = run_dir / "progress.json"
    if not path.exists():
        raise SystemExit(f"no run in {run_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    age = time.time() - data["updated"]
    print(f"{data['state']}, last update {age:.0f} s ago" if age < 60 else f"{data['state']}, last update {hours_minutes(age)} ago")
    for s in data["stages"]:
        print(f"  {s['label_en']:<26} {s['items_done']:>9,} of {s['items']:<9,} {s['unit']:<7} {s['status']}")
    if data["state"] == "running":
        finish = datetime.fromtimestamp(data["finish_at"])
        print(f"remaining about {hours_minutes(data['remaining_s'])}, expected finish {finish:%a %d.%m %H:%M}")
        if age > STALE_AFTER_S:
            print("no update for a while: the run may have stopped; the start command continues it")
    elif data["error"]:
        print(data["error"])


# ---- executors: run a stage's missing chunks and hand each finished chunk to on_done ----


def slim(stage: Stage, item: dict, dataset_root: Path) -> dict:
    if stage.kind == "extract":
        return {"image_id": item["image_id"], "path": image_path(item, dataset_root)}
    return {"pair_id": item["pair_id"], "probe_image_id": item["probe_image_id"], "reference_image_id": item["reference_image_id"]}


def _opencv_chunk(task: tuple) -> tuple:
    kind, index, items, features_dir = task
    started = time.time()
    if kind == "extract":
        results = [opencv_sift.extract_job((item["image_id"], item["path"], features_dir)) for item in items]
    else:
        results = [opencv_sift.compare_job((item, features_dir)) for item in items]
    return index, results, started, time.time()


def run_opencv(stage: Stage, todo: list, ctx: Context, on_done, tick) -> None:
    features = str(ctx.features_dir(stage.algorithm))
    tasks = [(stage.kind, index, [slim(stage, item, ctx.dataset_root) for item in chunk], features) for index, chunk in todo]
    with Pool(ctx.workers, initializer=opencv_sift.init_worker, initargs=(ctx.opencv_seed,)) as pool:
        finished = pool.imap_unordered(_opencv_chunk, tasks)
        for _ in tasks:
            while True:
                try:
                    index, results, started, ended = finished.next(timeout=1)
                    break
                except PoolTimeout:
                    tick()
            on_done(index, results, started, ended)
            tick()


def run_pore(stage: Stage, todo: list, ctx: Context, on_done, tick) -> None:
    keys = {(f"{stage.design}-" if stage.design else "") + f"{index:05d}": index for index, _ in todo}
    chunks = [(key, [slim(stage, item, ctx.dataset_root) for item in chunk]) for key, (_, chunk) in zip(keys, todo)]
    pore_sift.run_chunks(
        stage.kind, chunks, ctx.workers, ctx.pore_settings, ctx.features_dir(stage.algorithm),
        ctx.run_dir / "work" / stage.algorithm,
        on_done=lambda key, results, started, ended: on_done(keys[key], results, started, ended), tick=tick,
    )


class StopRequested(Exception):
    """The researcher asked the run to stop (--stop)."""


def execute(ctx: Context, stages: list[Stage], progress: Progress, executors: dict | None = None, after_stage=lambda stage: None) -> None:
    executors = executors or {"opencv-sift": run_opencv, "pore-sift": run_pore}
    stop_file = ctx.run_dir / "stop-requested"

    def tick() -> None:
        if stop_file.exists():
            raise StopRequested
        progress.write()

    for stage in stages:
        chunks = stage.chunks()
        todo = [(index, chunk) for index, chunk in enumerate(chunks) if index not in progress.done[stage.name]]
        if todo:
            progress.current = stage.name
            progress.write(force=True)
            log(ctx.run_dir, f"{stage.name}: {len(todo)} of {len(chunks)} chunks to run")

            def on_done(index, results, started, ended, stage=stage, chunks=chunks):
                progress.record(stage, save_chunk(ctx.run_dir, stage, index, chunks[index], results, started, ended))

            executors[stage.algorithm](stage, todo, ctx, on_done, tick)
            log(ctx.run_dir, f"{stage.name}: done")
        after_stage(stage)
    progress.current = None


# ---- results ----


def far_label(far: float) -> str:
    return f"{far * 100:g}%"


def result_tables(pairs: list[dict], scores: dict[str, dict], targets: tuple[float, ...]) -> dict:
    """TAR, FRR and FAR with pair counts per scenario (R6) at each target FAR (E2)."""
    groups: dict[str, dict] = {}
    for p in pairs:
        groups.setdefault(p["scenario"], {"genuine": [], "impostor": []})[p["kind"]].append(scores[p["pair_id"]]["score"])
    return {
        scenario: {
            "failures": sum(1 for s in g["genuine"] + g["impostor"] if s is None),
            "at_far": {far_label(t): tar_at_far(g["genuine"], g["impostor"], t) for t in targets},
        }
        for scenario, g in sorted(groups.items())
    }


def write_scores(path: Path, pairs: list[dict], compact_ids: set[str], scores: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(pairs, key=lambda p: (p["scenario"], p["frgp"], p["probe_image_id"], p["reference_image_id"])):
        r = scores[p["pair_id"]]
        rows.append({**{k: p[k] for k in SCORE_FIELDS[:6]}, "compact": int(p["pair_id"] in compact_ids),
                     "status": r["status"], "reason": r["reason"], "score": r["score"]})
    write_csv(path, SCORE_FIELDS, rows)


def runtime(progress: Progress, algorithm: str) -> dict:
    """Wall-clock view of each stage; official timings are the single-thread pilots (E5)."""
    out: dict = {"workers": progress.workers}
    for s in progress.stages:
        metas = list(progress.done[s.name].values())
        if s.algorithm == algorithm and metas:
            out[s.name] = {
                "items": sum(m["count"] for m in metas),
                "worker_seconds": round(sum(m["finished"] - m["started"] for m in metas), 1),
                "first_start": datetime.fromtimestamp(min(m["started"] for m in metas)).isoformat(timespec="seconds"),
                "last_finish": datetime.fromtimestamp(max(m["finished"] for m in metas)).isoformat(timespec="seconds"),
            }
    return out


def finish_stage(ctx: Context, stages: list[Stage], progress: Progress, stage: Stage) -> None:
    """After an algorithm's compact pairs: its compact results. After its other pairs: its full
    results and its scores file. A failure here does not stop the run: the scores are in the chunk
    files, and the next start writes the missing results."""
    try:
        write_results(ctx, stages, progress, stage)
    except Exception:
        log(ctx.run_dir, f"{stage.name}: results not written:\n" + traceback.format_exc())
        progress.warnings.append(f"התוצאות של {stage.label()} לא נכתבו בגלל שגיאה; הפרטים ב-run.log.")


def write_results(ctx: Context, stages: list[Stage], progress: Progress, stage: Stage) -> None:
    if stage.kind != "compare":
        return
    path = ctx.run_dir / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    entry = report["algorithms"].setdefault(stage.algorithm, {})
    compact = next(s for s in stages if (s.algorithm, s.design) == (stage.algorithm, "compact"))
    scores_path = ctx.run_dir / "scores" / f"{stage.algorithm}.csv"
    if stage.design == "compact":
        if "compact" in entry:
            return
        entry["compact"] = result_tables(stage.items, stage_results(ctx.run_dir, stage), FAR_TARGETS["compact"])
    else:
        if "full" in entry and scores_path.exists():
            return
        scores = stage_results(ctx.run_dir, compact) | stage_results(ctx.run_dir, stage)
        pairs = compact.items + stage.items
        entry["full"] = result_tables(pairs, scores, FAR_TARGETS["full"])
        write_scores(scores_path, pairs, {p["pair_id"] for p in compact.items}, scores)
    entry["runtime"] = runtime(progress, stage.algorithm)
    write_json(path, report)
    progress.results = report["algorithms"]
    log(ctx.run_dir, f"{stage.name}: results written to report.json")


# ---- guards for a long unattended run ----


def hold_lock(run_dir: Path):
    """One runner per run directory; the operating system drops the lock with the process."""
    handle = open(run_dir / "runner.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another run is already working in {run_dir}") from None
    return handle


_JOB = None


def end_workers_with_this_process() -> None:
    """Windows: worker processes must not outlive the run, or a rerun would race them over the same
    files. A job object that closes with this process takes every worker with it."""
    global _JOB
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not (job and kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
            and kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())):
        raise OSError(ctypes.get_last_error(), "cannot tie the worker processes to this run")
    _JOB = job  # stays open until the process ends


def keep_awake() -> None:
    """Windows: no sleep while the run works (the screen may still turn off); ends with the process."""
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def check_disk(run_dir: Path, stages: list[Stage], progress: Progress) -> None:
    needed_mb = 2000.0  # chunk files, scores and a margin
    for s in stages:
        if s.kind == "extract":
            left = len(s.items) - sum(m["count"] for m in progress.done[s.name].values())
            needed_mb += left * FEATURE_MB[s.algorithm]
    free_mb = shutil.disk_usage(run_dir).free / 1e6
    if free_mb < needed_mb:
        raise SystemExit(f"not enough disk space in {run_dir}: {free_mb / 1000:.0f} GB free, about {needed_mb / 1000:.0f} GB needed")


def git_state() -> str:
    try:
        head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head.strip() + (" with uncommitted changes" if dirty.strip() else "")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "full")
    parser.add_argument("--sample", help="run a saved sample instead (manifests/<name>_images.csv and _pairs.csv)")
    parser.add_argument("--status", action="store_true", help="print the progress and the expected finish")
    parser.add_argument("--stop", action="store_true", help="stop a running run; the start command continues it")
    args = parser.parse_args(argv)
    run_dir = args.out.resolve()
    if args.status:
        print_status(run_dir)
        return
    if args.stop:
        if not (run_dir / "progress.json").exists():
            raise SystemExit(f"no run in {run_dir}")
        (run_dir / "stop-requested").touch()
        print("stop requested: the run stops within a few seconds; the start command continues it")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    lock = hold_lock(run_dir)
    end_workers_with_this_process()
    keep_awake()
    (run_dir / "stop-requested").unlink(missing_ok=True)
    protocol, algorithms = load_toml("protocol.toml"), load_toml("algorithms.toml")
    settings = pore_sift.resolve(algorithms["pore_sift"], REPO_ROOT)
    pore_sift.check_upstream(settings)
    ctx = Context(run_dir, Path(protocol["dataset"]["root"]), args.workers,
                  opencv_sift.rng_seed(protocol["randomness"]["master_seed"]), settings)
    images, compact, rest = sample_design(args.sample) if args.sample else full_design(read_images(), protocol)
    stages = build_stages(images, compact, rest)
    plan = plan_record(stages, ctx, protocol, algorithms)
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8"))["plan"] != plan:
            raise SystemExit(f"{run_dir} holds a run with other inputs; choose another --out")
    else:
        write_json(plan_path, {"plan": plan, "created": time.time()})
        design = {"images": len(images), **{name: dict(sorted(Counter(f"{p['scenario']} {p['kind']}" for p in pairs).items()))
                                            for name, pairs in (("compact_pairs", compact), ("other_pairs", rest))}}
        write_json(run_dir / "report.json", {"design": design, "far_rule": FAR_RULE, "host": host(), "algorithms": {}})
    for algorithm in NAMES:
        ctx.features_dir(algorithm).mkdir(parents=True, exist_ok=True)
    progress = Progress(run_dir, stages, args.workers, json.loads(plan_path.read_text(encoding="utf-8"))["created"])
    progress.results = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["algorithms"]
    check_disk(run_dir, stages, progress)
    done = sum(len(d) for d in progress.done.values())
    log(run_dir, f"start: {args.workers} workers, code {git_state()}, {done} of {sum(s.n_chunks for s in stages)} chunks already done")
    try:
        execute(ctx, stages, progress, after_stage=lambda stage: finish_stage(ctx, stages, progress, stage))
        progress.state = "finished"
    except (StopRequested, KeyboardInterrupt):
        progress.state = "stopped"
        (run_dir / "stop-requested").unlink(missing_ok=True)
    except BaseException as error:
        progress.state, progress.error = "failed", f"{type(error).__name__}: {error}"
        log(run_dir, "failed:\n" + traceback.format_exc())
        raise
    finally:
        progress.write(force=True)
        log(run_dir, progress.state)
        lock.close()


if __name__ == "__main__":
    main()
