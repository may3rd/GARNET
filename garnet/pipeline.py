"""Deterministic driver: every non-LLM stage and every gate, no model calls.

Where a stage needs an LLM agent it writes dispatch.json and exits 3; the orchestrator fulfils
the requests and reruns with --resume. Exit: 0 pass, 1 hard gate failed, 2 bad input, 3 waiting.

    python3 -m garnet.pipeline --job demo --sheets <stem> [<stem> ...] --from-fixtures garnet/tests/input --no-agents
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

from garnet.graph.build_graph import build_graph
from garnet.graph.contracts import parse_line_number, short_sheet
from garnet.graph.link_sheets import link_sheets
from garnet.stage5b.host import _equipment_as_objects

REPO = Path(__file__).resolve().parents[1]
MAX_ATTEMPTS = 2
CONNECTORS = {"page connection", "utility connection"}
# fixture spellings in load_fixtures precedence order; staged under the first one
SPELL = {"A": ["{s}_objects.json"],
         "B": ["{s}_equipment_bboxes.json", "{s}_equipment.json"],
         "C": ["{s}_line_number_boxes.json", "{s}_line_numbers.json"]}
STAGED = {"A": "{s}_objects.json", "B": "{s}_equipment_bboxes.json", "C": "{s}_line_numbers.json"}


def _j(p):
    return json.loads(Path(p).read_text())


def _dump(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- gates (callable on their own)
def check_frame(sdir: Path, stem: str) -> tuple[bool, str]:
    """G0: raster exists and every declared JSON frame equals its size."""
    img = sdir / f"{stem}.jpg"
    im = cv2.imread(str(img)) if img.is_file() else None
    if im is None:
        return False, f"raster {img.name} missing or unreadable"
    h, w = im.shape[:2]
    bad = []
    for key in "ABC":
        p = sdir / STAGED[key].format(s=stem)
        if not p.is_file():
            continue
        d = _j(p)
        cf = d.get("coordinate_frame") or {}
        fw, fh = cf.get("width_px") or d.get("image_width"), cf.get("height_px") or d.get("image_height")
        # ponytail: A also checked via image_width/height; a field that is absent or 0 is skipped
        if (fw and int(fw) != w) or (fh and int(fh) != h):
            bad.append(f"{p.name} declares {fw}x{fh}")
    return (not bad), (f"raster is {w}x{h} but " + "; ".join(bad)) if bad else f"{w}x{h}"


def check_equipment(sdir: Path, stem: str) -> tuple[bool, str, dict]:
    """G1: >=1 equipment item survives the host vocabulary; every port has side + 2-element point_px."""
    p = sdir / STAGED["B"].format(s=stem)
    if not p.is_file():
        return False, f"no equipment file {p.name}", {}
    items = _j(p).get("objects") or []
    kept = [it for it in items if _equipment_as_objects([it])]
    dropped = [f"{it.get('Tag') or '?'} ({it.get('Equipment_type')!r})" for it in items if it not in kept]
    badports = []
    for it in items:
        for pt in it.get("Ports") or []:
            xy = pt.get("point_px")
            if isinstance(xy, dict):
                xy = (xy.get("x"), xy.get("y"))
            if not pt.get("side") or not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                badports.append(f"{it.get('Tag')}:{pt.get('mark')}")
    info = {"items": len(items), "kept": len(kept), "dropped": dropped, "bad_ports": badports}
    why = []
    if not kept:
        why.append("0 equipment items map into the vocabulary")
    if dropped:
        why.append("dropped by vocabulary: " + ", ".join(dropped))
    if badports:
        why.append("ports without side/point_px: " + ", ".join(badports))
    return (bool(kept) and not badports), "; ".join(why) or f"{len(kept)} items", info


def check_text(sdir: Path, stem: str) -> tuple[bool, str, dict]:
    """G2 (soft): connectors carry text; line numbers parse. Returns failing ids per agent role."""
    fail = {"instrument_reader": [], "line_numbers": []}
    a = sdir / STAGED["A"].format(s=stem)
    for o in (_j(a).get("objects") or []) if a.is_file() else []:
        if str(o.get("class_name", "")).lower() in CONNECTORS and not str(o.get("text") or o.get("Text") or "").strip():
            fail["instrument_reader"].append(o["id"])
    c = sdir / STAGED["C"].format(s=stem)
    for o in (_j(c).get("objects") or []) if c.is_file() else []:
        if not parse_line_number(str(o.get("Text") or "")):
            fail["line_numbers"].append(f"line_number_{o.get('Index')}:{str(o.get('Text') or '')!r}")
    fail = {k: v for k, v in fail.items() if v}
    return (not fail), "; ".join(f"{k}: {len(v)} failing" for k, v in fail.items()) or "ok", fail


def claim_attempt(report: dict, item: dict) -> bool:
    """Count a dispatch; False (-> hard failure) when the role is already at its cap."""
    n = report["attempts"].get(item["id"], 0)
    if n >= item["max_attempts"]:
        return False
    report["attempts"][item["id"]] = item["attempt"] = n + 1
    return True


def apply_decisions(graphs: dict, decisions: list) -> list:
    """reject edge -> remove; confirm -> evidence.review=confirmed; fix -> merge fields."""
    applied = []
    for d in decisions:
        t, kind = d.get("target"), d.get("decision")
        for g in graphs.values():
            hit = next((x for x in g["edges"] if x["id"] == t), None) or next((x for x in g["nodes"] if x["id"] == t), None)
            if not hit:
                continue
            is_edge = "u" in hit
            if kind == "reject" and is_edge:
                g["edges"].remove(hit)
                for r in g.get("line_list", []):
                    r["edges"] = [e for e in r["edges"] if e != t]
            elif kind == "confirm":
                hit.setdefault("evidence", {})["review"] = "confirmed"
            elif kind == "fix":
                hit.update(d.get("fix") or {})
            else:
                break  # ponytail: reject on a node is not supported, left unapplied
            applied.append(f"{kind} {t}")
            break
    return applied


# ---------------------------------------------------------------- stages
def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run([sys.executable, "-m", *cmd], cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode


def intake(sdir: Path, stem: str, fixtures: Path | None, images: Path | None, resume: bool):
    sdir.mkdir(parents=True, exist_ok=True)
    src = fixtures or images
    jpg = sdir / f"{stem}.jpg"
    if src and (src / f"{stem}.jpg").is_file() and not (resume and jpg.exists()):
        shutil.copy(src / f"{stem}.jpg", jpg)
    if fixtures:
        for k, names in SPELL.items():
            dst = sdir / STAGED[k].format(s=stem)
            if resume and dst.exists():
                continue
            for n in names:
                if (fixtures / n.format(s=stem)).is_file():
                    shutil.copy(fixtures / n.format(s=stem), dst)
                    break


def process_sheet(a, job: Path, stem: str, rep: dict, pending: list) -> dict | None:
    """Return the sheet's graph or None; fills rep['sheets'][stem] with gate results."""
    sdir, odir = job / "sheets" / stem, job / "out" / stem
    S = rep["sheets"].setdefault(stem, {"gates": {}})
    G = S["gates"]
    can_dispatch = not (a.from_fixtures or a.no_agents)

    def gate(name, status, reason="", t0=None, **extra):
        G[name] = {"status": status, "reason": reason, "seconds": round(time.time() - t0, 1) if t0 else 0, **extra}
        return status

    def fail(name, reason):  # hard failure for the sheet
        gate(name, "fail", reason)
        S["status"] = "fail"

    def dispatch(role, reason, inputs=None, output=None):
        item = {"id": f"{stem}:{role}", "role": role, "sheet": stem, "image": str(sdir / f"{stem}.jpg"),
                "inputs": inputs or {k: str(sdir / STAGED[k].format(s=stem)) for k in "ABC"},
                "output": str(output or sdir / STAGED[{"equipment": "B", "line_numbers": "C"}.get(role, "A")].format(s=stem)),
                "reason": reason, "attempt": 0, "max_attempts": MAX_ATTEMPTS}
        if claim_attempt(rep, item):
            pending.append(item)
            return True
        return False

    intake(sdir, stem, a.from_fixtures, a.images, a.resume)
    if not (sdir / f"{stem}_objects.json").is_file():
        # ponytail: local YOLO only (no LLM); weights come from detect_objects defaults
        run(["garnet.detect_objects", "--image", str(sdir / f"{stem}.jpg"), "--out", str(sdir / f"{stem}_objects.json")], job / f"{stem}_detect.log")
    t = time.time()
    ok, why = check_frame(sdir, stem)
    if not (sdir / f"{stem}_objects.json").is_file():
        ok, why = False, f"no {stem}_objects.json"
    if not ok:
        fail("G0", why)
        S["bad_input"] = not (sdir / f"{stem}.jpg").is_file() or why.startswith("no ")
        return None
    gate("G0", "pass", why, t)

    t = time.time()
    ok, why, info = check_equipment(sdir, stem)
    if not ok:
        if can_dispatch and dispatch("equipment", why):
            gate("G1", "fail", "dispatched: " + why, t, **info)
            S["waiting"] = True
        else:
            fail("G1", why + ("" if can_dispatch else " (frozen inputs, no dispatch)"))
            G["G1"].update(info)
        return None
    gate("G1", "pass", why, t, **info)

    t = time.time()
    has_c = (sdir / STAGED["C"].format(s=stem)).is_file()
    if not has_c and not a.no_agents and not a.from_fixtures:
        # ponytail: line numbers are only dispatched when never supplied; a missing C under fixtures runs without them
        if dispatch("line_numbers", "no line number file"):
            gate("G2", "warn", "dispatched line_numbers: file missing", t)
            S["waiting"] = True
            return None
        fail("G2", "line_numbers attempts exhausted")
        return None
    ok, why, fails = check_text(sdir, stem)
    if ok:
        gate("G2", "pass", why, t)
    else:
        sent = []
        if not a.no_agents:
            for role, ids in fails.items():
                if dispatch(role, f"{len(ids)} failing: {ids}"):
                    sent.append(role)
                else:
                    fail("G2", f"{role} attempts exhausted: {ids}")
                    return None
        gate("G2", "warn", why, t, failing=fails, dispatched=sent)
        if sent:
            S["waiting"] = True
            return None

    t = time.time()
    # --resume skips a tracer whose stage6 output already exists
    s6 = odir / "stage6_trace_associations.json"
    if not (a.resume and s6.is_file()):
        args = ["--stem", stem]
        for c in (["garnet.stage5b.run", *args, "--indir", str(sdir), "--out", str(odir)],
                  ["garnet.stage5b.review_gate", *args, "--dir", str(odir), "--no-llm"],
                  ["garnet.stage5b.run_associations", *args, "--indir", str(sdir), "--dir", str(odir)]):
            odir.mkdir(parents=True, exist_ok=True)
            rc = run(c, odir / f"{c[0].rsplit('.', 1)[-1]}.log")
            if rc:
                gate("G3", "fail", f"{c[0]} exit {rc} (see {c[0].rsplit('.', 1)[-1]}.log)", t)
                if not s6.is_file():
                    S["status"] = "fail"
                    return None
    if not s6.is_file():
        fail("G3", "no stage6 output")
        return None
    d = _j(s6)
    un = d.get("unresolved") or {}
    n = len(d.get("trace_edges") or [])
    cnt = {"traced": n, "skipped_branches": len(un.get("skipped_branches") or []),
           "dead_end_traces": len(un.get("dead_end_traces") or []),
           "traces_without_line_number": len(un.get("traces_without_line_number") or [])}
    gate("G3", "pass" if n else "fail", "" if n else "stage6 has 0 trace edges", t, **cnt)
    if not n:
        S["status"] = "fail"
        return None

    # G4: deterministic candidates, then a vision agent picks the pipe each label belongs to
    t = time.time()
    cj, aj = odir / "ln_candidates.json", odir / "ln_assign.json"
    if not (a.resume and cj.is_file()):
        run(["garnet.graph.ln_candidates", "--stem", stem, "--dir", str(odir), "--indir", str(sdir)], odir / "ln_candidates.log")
    labels = [l["id"] for l in (_j(cj).get("labels") or [])] if cj.is_file() else []
    if not cj.is_file():
        gate("G4", "warn", "ln_candidates produced no json; geometric line-number assignment only", t)
    elif not labels:
        gate("G4", "pass", "no labels", t)
    elif not aj.is_file():
        if a.no_agents:
            gate("G4", "warn", "geometric line-number assignment only", t)
        elif dispatch("ln_assign", f"{len(labels)} labels need a vision pick", {"candidates": str(cj), "image": str(sdir / f"{stem}.jpg")}, aj):
            gate("G4", "warn", "dispatched ln_assign", t)
            S["waiting"] = True
            return None  # build_graph would bake in the geometric pick
        else:
            fail("G4", "ln_assign attempts exhausted")
            return None
    else:
        got = {x.get("id"): x for x in _j(aj).get("assignments") or []}
        bad = [l for l in labels if l not in got or got[l].get("choice") == "none" or not got[l].get("trace_id")]
        gate("G4", "warn" if bad else "pass", f"{len(bad)}/{len(labels)} none/undecided/missing: {bad[:10]}" if bad else f"{len(labels)} assigned", t)

    t = time.time()
    try:
        g = build_graph(stem, odir, sdir)
    except Exception as ex:  # polyline jump etc.
        gate("G5", "warn", f"build_graph raised: {ex}", t)
        return None
    _dump(odir / "graph.json", g)
    pipes = [e for e in g["edges"] if e["kind"] == "pipe"]
    kinds: dict = {}
    for f in g["flags"]:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    gate("G5", "warn" if g["flags"] else "pass", f"{len(g['flags'])} flags", t,
         pipes=len(pipes), pipes_without_line_core=sum(1 for e in pipes if not e.get("line_core")), flags=kinds)
    return g


def load_decisions(job: Path) -> list:
    out = []
    for p in [job / "review" / "review_decisions.json", *sorted((job / "review").glob("*_decisions.json"))]:
        if p.is_file():
            d = _j(p)
            out += d if isinstance(d, list) else d.get("decisions", [])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True)
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--from-fixtures", type=Path)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--golden", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-agents", action="store_true")
    ap.add_argument("--root", default="output")
    a = ap.parse_args(argv)
    root = Path(a.root)
    job = (root if root.is_absolute() else REPO / root) / a.job
    for attr in ("from_fixtures", "images", "golden"):
        v = getattr(a, attr)
        if v is not None:
            setattr(a, attr, v.resolve())
            if not v.is_dir():
                print(f"bad input: {v} is not a directory", file=sys.stderr)
                return 2
    job.mkdir(parents=True, exist_ok=True)
    rp = job / "run_report.json"
    rep = _j(rp) if a.resume and rp.is_file() else {}
    rep = {"job": a.job, "sheets": {}, "attempts": rep.get("attempts", {})}
    pending: list = []
    graphs: dict = {}
    for stem in a.sheets:
        g = process_sheet(a, job, stem, rep, pending)
        if g:
            graphs[stem] = g
    S = rep["sheets"]
    if any(s.get("bad_input") for s in S.values()):
        _finish(job, rep, pending, "bad_input")
        return 2

    decisions = load_decisions(job)
    rep["review"] = {"decisions": len(decisions), "applied": apply_decisions(graphs, decisions) if decisions else []}
    if decisions:
        for stem, g in graphs.items():
            _dump(job / "out" / stem / "graph.json", g)
    elif not a.no_agents:
        for stem, g in graphs.items():
            if g["flags"]:
                item = {"id": f"{stem}:review", "role": "review", "sheet": stem, "image": str(job / "sheets" / stem / f"{stem}.jpg"),
                        "inputs": {"graph": str(job / "out" / stem / "graph.json"), "overlay": str(job / "out" / stem / "stage6_trace_association_overlay.png")},
                        "output": str(job / "review" / f"{stem}_decisions.json"),
                        "reason": f"{len(g['flags'])} flags: {sorted({f['kind'] for f in g['flags']})}", "attempt": 0, "max_attempts": MAX_ATTEMPTS}
                if claim_attempt(rep, item):
                    pending.append(item)
                    S[stem]["waiting"] = True
                else:
                    S[stem]["gates"]["G5"]["reason"] += "; review attempts exhausted"
    if pending:
        _finish(job, rep, pending, "waiting")
        return 3

    status = "pass"
    if graphs:
        t = time.time()
        merged = link_sheets(list(graphs.values()))
        _dump(job / "graph_merged.json", merged)
        cs = merged["cross_sheet"]
        rep["links"] = {"count": len(cs["links"]), "unmatched": cs["unmatched"],
                        "cores": [e.get("line_core") for e in cs["links"]], "seconds": round(time.time() - t, 1)}
        if a.golden:
            from garnet.eval.score import score
            gs = [_j(p) for p in sorted(a.golden.glob("*.json")) if p.stem in a.sheets]
            res = score(merged, gs)
            _dump(job / "eval" / "score.json", res)
            rep["score"] = {"pass": res["pass"], "metrics": {k: m["value"] for k, m in res["metrics"].items()}}
            if not res["pass"]:
                status = "fail"
            for stem in graphs:
                gf = a.golden / f"{stem}.json"
                if gf.is_file():
                    out = job / "eval" / "review" / short_sheet(stem)
                    run(["garnet.eval.render_golden", "--stem", stem, "--golden", str(gf), "--graph", str(job / "out" / stem / "graph.json"),
                         "--image", str(job / "sheets" / stem / f"{stem}.jpg"), "--out", str(out)], job / "eval" / f"render_{stem}.log")
                    if (out / "index.html").is_file():
                        rep.setdefault("review_index", []).append(str(out / "index.html"))
    if any(s.get("status") == "fail" for s in S.values()):
        status = "fail"
    _finish(job, rep, pending, status)
    return 0 if status == "pass" else 1


def _finish(job, rep, pending, status):
    rep["status"] = status
    rep["dispatched"] = [p["id"] for p in pending]
    _dump(job / "run_report.json", rep)
    if pending:
        _dump(job / "dispatch.json", {"job": rep["job"], "pending": pending})
    elif (job / "dispatch.json").exists():
        try:
            (job / "dispatch.json").unlink()
        except OSError:  # ponytail: read-only/undeletable mount, blank the file instead
            _dump(job / "dispatch.json", {"job": rep["job"], "pending": []})
    print(f"job {rep['job']}: {status}")
    for stem, s in rep["sheets"].items():
        for gname, g in s["gates"].items():
            print(f"  {stem.rsplit('-', 1)[-1]:>6} {gname:3} {g['status']:5} {g['seconds']:>5}s  {g['reason'][:90]}")
    if "links" in rep:
        print(f"  links {rep['links']['count']} {rep['links']['cores']}, unmatched {len(rep['links']['unmatched'])}")
    for p in rep.get("review_index", []):
        print(f"  golden review {p}")
    if "score" in rep:
        print(f"  score pass={rep['score']['pass']}")
    for p in pending:
        print(f"  dispatch {p['id']}: {p['reason'][:80]}")


if __name__ == "__main__":
    sys.exit(main())
