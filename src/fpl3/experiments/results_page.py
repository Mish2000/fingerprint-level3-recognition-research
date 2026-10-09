"""One results page for the researcher and the supervisor: every algorithm and method run so far, in
the progress page's style, results only (no explanations).

    python -m fpl3.experiments.results_page        -> runs/results/results.html

Reads runs/full/report.json, runs/method/fusion/*/report.json and the F1 sample reports; the page
is self-contained (one file, no scripts, no external resources) so it can be sent as is.
"""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

from .common import REPO_ROOT
from .full_run import PAGE_CSS, isolate_names, results_html

OUT = REPO_ROOT / "runs" / "results" / "results.html"
FUSION = REPO_ROOT / "runs" / "method" / "fusion"
FARS = ("1%", "0.1%", "0.01%")
SCENARIOS = (("UxV", "מגולגלת מול מגולגלת"), ("RxV", "שטוחה מול מגולגלת"))

# (group, row label, fusion report, algorithm key in it)
ROWS = (
    ("אלגוריתמים קיימים", "OpenCV SIFT", "all-three", "opencv-sift"),
    ("אלגוריתמים קיימים", "Pore SIFT", "all-three", "pore-sift"),
    ("אלגוריתמים קיימים", "שילוב OpenCV SIFT ו-Pore SIFT", "reference", "fusion"),
    ("מודל נלמד · אימון 1", "לבד", "all-three", "learned-pore"),
    ("מודל נלמד · אימון 1", "עם Pore SIFT", "learned-pore-sift", "fusion"),
    ("מודל נלמד · אימון 1", "עם OpenCV SIFT", "learned-opencv", "fusion"),
    ("מודל נלמד · אימון 1", "עם OpenCV SIFT ו-Pore SIFT", "all-three", "fusion"),
    ("מודל נלמד · אימון 2", "לבד", "repeat-all-three", "learned-pore-repeat"),
    ("מודל נלמד · אימון 2", "עם Pore SIFT", "repeat-learned-pore-sift", "fusion"),
    ("מודל נלמד · אימון 2", "עם OpenCV SIFT", "repeat-learned-opencv", "fusion"),
    ("מודל נלמד · אימון 2", "עם OpenCV SIFT ו-Pore SIFT", "repeat-all-three", "fusion"),
)
F1_REPORTS = (("OpenCV SIFT", "f1-opencv-sift-seeded"), ("Pore SIFT", "f1-pore-sift-rerun"), ("scikit-image Harris", "f1-skimage-harris"))

EXTRA_CSS = """
.group td { font-weight: 600; color: var(--muted); background: var(--accent-soft); font-size: .85rem; }
.best { font-weight: 700; }
.range { display: block; color: var(--muted); font-size: .75rem; font-weight: 400; }
details { border-top: 1px solid var(--line); padding: 8px 0; }
details summary { cursor: pointer; font-weight: 600; }
"""


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_results() -> dict[tuple[str, str, str], dict]:
    """(group, row label, scenario) -> the algorithm entry: at_far points with tar_ci."""
    reports = {name: load(FUSION / name / "report.json") for name in {r[2] for r in ROWS}}
    return {(group, label, scenario): reports[report]["scenarios"][scenario]["algorithms"][key]
            for group, label, report, key in ROWS for scenario, _ in SCENARIOS}


def percent(x: float) -> str:
    return f"{100 * x:.1f}%"


def summary_table(results: dict, scenario: str) -> str:
    best = {far: max(results[(g, l, scenario)]["at_far"][far]["tar"] for g, l, _, _ in ROWS) for far in FARS}
    rows, group = [], None
    for g, label, _, _ in ROWS:
        if g != group:
            rows.append(f'<tr class="group"><td colspan="4">{isolate_names(html.escape(g))}</td></tr>')
            group = g
        cells = []
        for far in FARS:
            r = results[(g, label, scenario)]["at_far"][far]
            low, high = r["tar_ci"]
            cls = "num best" if r["tar"] == best[far] else "num"
            cells.append(f'<td class="{cls}">{percent(r["tar"])}'
                         f'<span class="range"><bdi dir="ltr">[{100 * low:.1f}–{100 * high:.1f}]</bdi></span></td>')
        rows.append(f"<tr><td>{isolate_names(html.escape(label))}</td>{''.join(cells)}</tr>")
    head = "".join(f'<th><bdi>TAR</bdi> ב-<bdi dir="ltr">FAR {far}</bdi></th>' for far in FARS)
    return f'<div class="table-wrap"><table><thead><tr><th>שיטה</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def detail_table(entry_by_scenario: dict[str, dict]) -> str:
    rows = []
    for scenario, name in SCENARIOS:
        points = entry_by_scenario[scenario]["at_far"]
        for i, far in enumerate(FARS):
            r = points[far]
            head = f'<td rowspan="{len(FARS)}">{name}</td>' if i == 0 else ""
            rows.append(
                f'<tr>{head}<td class="num">{far}</td><td class="num strong">{percent(r["tar"])}</td>'
                f'<td class="num">{percent(r["frr"])}</td><td class="num">{r["true_accepts"]:,} מתוך {r["genuine"]:,}</td>'
                f'<td class="num">{r["false_accepts"]:,} מתוך {r["impostor"]:,}</td></tr>'
            )
    return ('<div class="table-wrap"><table><thead><tr><th>השוואה</th><th>FAR</th><th>TAR</th><th>FRR</th>'
            f'<th>זוגות אמת שהתקבלו</th><th>מתחזים שהתקבלו</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def f1_table() -> str:
    rows = []
    for name, folder in F1_REPORTS:
        results = load(REPO_ROOT / "runs" / folder / "report.json")["results"]
        cells = "".join(f'<td class="num">{results[s]["true_accepts"]} מתוך {results[s]["genuine"]}</td>' for s, _ in SCENARIOS)
        rows.append(f"<tr><td>{isolate_names(html.escape(name))}</td>{cells}</tr>")
    head = "".join(f"<th>{n}</th>" for _, n in SCENARIOS)
    return f'<div class="table-wrap"><table><thead><tr><th>אלגוריתם</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def render() -> str:
    results = test_results()
    full = load(REPO_ROOT / "runs" / "full" / "report.json")
    sample = results[(ROWS[0][0], ROWS[0][1], "UxV")]["at_far"]["1%"], results[(ROWS[0][0], ROWS[0][1], "RxV")]["at_far"]["1%"]
    details = "".join(
        f"<details><summary>{isolate_names(html.escape(g))} · {isolate_names(html.escape(label))}</summary>"
        f"{detail_table({s: results[(g, label, s)] for s, _ in SCENARIOS})}</details>"
        for g, label, _, _ in ROWS
    )
    return f"""<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>תוצאות המחקר</title>
<style>{PAGE_CSS}{EXTRA_CSS}</style>
</head>
<body>
<main>
<h1>תוצאות המחקר</h1>
<p class="muted">NIST SD302 · טביעות 1000 DPI מקוריות · עדכון {date.today():%d.%m.%Y}</p>

<section class="card"><h2>נבדקי הבדיקה (100 נבדקים)</h2>
<p class="muted">מגולגלת מול מגולגלת: {sample[0]["genuine"]:,} זוגות אמת ו-{sample[0]["impostor"]:,} מתחזים ·
שטוחה מול מגולגלת: {sample[1]["genuine"]:,} זוגות אמת ו-{sample[1]["impostor"]:,} מתחזים · בסוגריים: טווח ביטחון 95%</p>
<h3>מגולגלת מול מגולגלת</h3>{summary_table(results, "UxV")}
<h3>שטוחה מול מגולגלת</h3>{summary_table(results, "RxV")}
</section>

<section class="card"><h2>פירוט לכל שיטה, נבדקי הבדיקה</h2>{details}</section>

<section class="card"><h2>הרצה מלאה, כל 200 הנבדקים</h2>{results_html(full["algorithms"])}</section>

<section class="card"><h2>מדגם F1 (400 זוגות)</h2>
<p class="muted">זוגות אמת שהתקבלו בסף שבו אף אחד מ-180 המתחזים לא התקבל</p>{f1_table()}</section>
</main>
</body>
</html>
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(OUT)


if __name__ == "__main__":
    main()
