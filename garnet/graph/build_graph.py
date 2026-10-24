"""Stage 6 tracer output -> Contract D graph (garnet_graph_v1) for one sheet.

    python3 -m garnet.graph.build_graph --stem <S> --dir <tracer out dir> [--indir garnet/tests/input]

Node ids are semantic (`0002A/V-2501/AO`, `0002A/PC/<text>@1`), never detector ids; the detector id
rides along in `src_id`. Trace ends on another trace (`tee_junction`) split that trace at a tee node.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from garnet.graph.contracts import SCHEMA, line_core, parse_line_number, short_sheet

TEE_REACH = 20.0     # px: tee endpoint -> nearest point on another trace
TEE_MERGE = 12.0     # px: endpoints this close share one tee
MAX_STEP = 150.0     # px: densify long segments so the 200 px invariant means something
MAX_JUMP = 200.0
CONNECTORS = {  # object class -> (id prefix, node kind)
    "page connection": ("PC", "page_connection"),
    "utility connection": ("UC", "utility_connection"),
    "connection": ("CN", "connection"),
    "sampling point": ("SP", "sampling_point"),
    "pressure relief valve": ("PSV", "pressure_relief_valve"),
}
DEAD = {"dead_end", "no_pipe"}


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def _densify(pts):
    out = [tuple(map(float, pts[0]))]
    for p in pts[1:]:
        p = tuple(map(float, p))
        d = math.dist(out[-1], p)
        if d == 0:
            continue
        k = int(d // MAX_STEP)
        a = out[-1]
        for i in range(1, k + 1):
            f = i / (k + 1)
            out.append((a[0] + (p[0] - a[0]) * f, a[1] + (p[1] - a[1]) * f))
        out.append(p)
    return out


class Trace:
    def __init__(self, t):
        self.id, self.raw = t["trace_id"], t
        self.pts = _densify(t["polyline"])
        self.cum = [0.0]
        for a, b in zip(self.pts, self.pts[1:]):
            self.cum.append(self.cum[-1] + math.dist(a, b))
        self.L = self.cum[-1]
        self.cuts = []  # [s, node_id, node_xy]

    def project(self, xy):
        """-> (dist, s, point) of the nearest point on the polyline."""
        best = (1e18, 0.0, self.pts[0])
        for i in range(len(self.pts) - 1):
            a, b = self.pts[i], self.pts[i + 1]
            dx, dy = b[0] - a[0], b[1] - a[1]
            seg = dx * dx + dy * dy
            f = 0.0 if seg == 0 else max(0.0, min(1.0, ((xy[0] - a[0]) * dx + (xy[1] - a[1]) * dy) / seg))
            p = (a[0] + dx * f, a[1] + dy * f)
            d = math.dist(xy, p)
            if d < best[0]:
                best = (d, self.cum[i] + f * math.sqrt(seg), p)
        return best

    def point_at(self, s):
        for i in range(len(self.pts) - 1):
            if s <= self.cum[i + 1] or i == len(self.pts) - 2:
                seg = self.cum[i + 1] - self.cum[i]
                f = 0.0 if seg == 0 else max(0.0, min(1.0, (s - self.cum[i]) / seg))
                a, b = self.pts[i], self.pts[i + 1]
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)

    def piece(self, s0, s1, xy0, xy1):
        """Polyline from s0 to s1 in trace direction (always oriented u->v), ends snapped to node xy."""
        pts = [self.point_at(s0)] + [p for p, c in zip(self.pts, self.cum) if s0 < c < s1] + [self.point_at(s1)]
        if math.dist(pts[0], xy0) > 0.5:
            pts.insert(0, tuple(xy0))
        if math.dist(pts[-1], xy1) > 0.5:
            pts.append(tuple(xy1))
        return [[round(x, 1), round(y, 1)] for x, y in pts]


def _arc(poly):
    return sum(math.dist(a, b) for a, b in zip(poly, poly[1:]))


def build_graph(stem: str, out_dir, indir="garnet/tests/input") -> dict:
    out_dir, indir = Path(out_dir), Path(indir)
    s6 = _load(out_dir / "stage6_trace_associations.json")
    if s6 is None:
        raise FileNotFoundError(f"missing {out_dir / 'stage6_trace_associations.json'}")
    sh = short_sheet(stem)
    nodes, edges, flags = {}, [], []

    def flag(kind, where, why):
        flags.append({"kind": kind, "where": where, "why": why})

    def add_node(nid, kind, **kw):
        nodes[nid] = {"id": nid, "kind": kind, "sheet": sh, **kw}
        return nid

    # ---- objects (class + connector text) -------------------------------------------------
    objs = {o["id"]: o for o in ((_load(out_dir / "stage4_objects.json") or {}).get("objects") or [])}
    for o in ((_load(indir / f"{stem}_objects.json") or {}).get("objects") or []):
        objs.setdefault(o["id"], o)
        extra = {"text": o.get("text") or o.get("Text"), "direction": o.get("direction") or o.get("Direction")}
        objs[o["id"]] = {**objs[o["id"]], **{k: v for k, v in extra.items() if v}}

    # ---- equipment + ports ----------------------------------------------------------------
    eq_ports = _load(out_dir / "ai_equipment_ports.json") or {}
    port_node = {}  # (eq_id, index) -> port node id
    eq_tag = {}
    for e in ((_load(out_dir / "stage3_equipment_bboxes.json") or {}).get("equipment") or []):
        tag = e.get("tag") or e["id"]
        eid = add_node(f"{sh}/{tag}", "equipment", tag=tag, type=e.get("class_name", ""),
                       bbox_px=[e["bbox"][k] for k in ("x_min", "y_min", "x_max", "y_max")],
                       xy_px=[(e["bbox"]["x_min"] + e["bbox"]["x_max"]) / 2, (e["bbox"]["y_min"] + e["bbox"]["y_max"]) / 2],
                       src_id=e["id"])
        eq_tag[e["id"]] = eid
        for i, p in enumerate(eq_ports.get(e["id"], []), 1):
            mark = p.get("mark") or ""
            if not mark or re.fullmatch(r"port_\d+", mark):  # host.py fills port_NN when the fixture has no mark
                mark = f"p{i}"
            pid = f"{eid}/{mark}"
            if pid in nodes:
                pid = f"{pid}.{i}"
            add_node(pid, "port", equipment=eid, mark=mark, xy_px=[p["x"], p["y"]], src_id=f"{e['id']}:port_{i:02d}")
            port_node[(e["id"], i)] = pid
            edges.append({"id": f"{eid}->{pid}", "kind": "has_port", "u": eid, "v": pid})

    def resolve_port(eq_id, index, xy):
        # Stage 5 projects port xy onto the bbox edge, so it differs from the nozzle point by tens of px:
        # trust the 1-based port_index into ai_equipment_ports, fall back to nearest nozzle within 60 px.
        pl = eq_ports.get(eq_id, [])
        if index and 1 <= index <= len(pl):
            return port_node.get((eq_id, index))
        near = sorted((math.dist(xy, (p["x"], p["y"])), i) for i, p in enumerate(pl, 1))
        return port_node.get((eq_id, near[0][1])) if near and near[0][0] <= 60 else None

    # ---- traces ---------------------------------------------------------------------------
    traces = []
    for t in s6["trace_edges"]:
        poly = t.get("polyline") or []
        if len(poly) < 2 or t.get("status") not in ("ok", "traced", None):
            flag("trace_dropped", t["trace_id"], f"polyline points={len(poly)} status={t.get('status')}")
            continue
        tr = Trace(t)
        if tr.L < 1:
            flag("trace_dropped", t["trace_id"], "zero length polyline")
            continue
        traces.append(tr)

    # connector nodes, numbered per (prefix, text) in reading order so ids do not depend on detector ids
    conn_refs = {}  # obj_id -> (class, xy)
    for tr in traces:
        t = tr.raw
        if t["source_obj_type"] in CONNECTORS:
            conn_refs.setdefault(t["source_obj_id"], (t["source_obj_type"], tr.pts[0]))
        tid = t.get("terminal_obj_id")
        if tid:
            cls = (objs.get(tid) or {}).get("class_name", "")
            if cls in CONNECTORS and t["terminal_type"] not in ("instrument_tag", "equipment"):
                conn_refs.setdefault(tid, (cls, tuple(t["terminal_xy"])))
    seen = Counter()
    conn_node = {}

    def _pos(oid):
        b = (objs.get(oid) or {}).get("bbox") or {}
        return (b.get("y_min", 0), b.get("x_min", 0), oid)

    for oid in sorted(conn_refs, key=_pos):
        cls, xy = conn_refs[oid]
        prefix, kind = CONNECTORS[cls]
        text = str((objs.get(oid) or {}).get("text") or "").strip()
        if not text:
            text = oid
            flag("no_connector_text", oid, f"{cls} has no text; id uses detector id")
        direction = str((objs.get(oid) or {}).get("direction") or "").strip().lower()
        direction = direction if direction in ("to", "from") else ""
        seen[(prefix, text)] += 1
        conn_node[oid] = add_node(f"{sh}/{prefix}/{text}@{seen[(prefix, text)]}", kind, text=text,
                                  direction=direction, xy_px=[xy[0], xy[1]], src_id=oid)

    end_n = [0]

    def end_node(xy, tid, terminal_type, src=None):
        end_n[0] += 1
        n = add_node(f"{sh}/END{end_n[0]}", "end", xy_px=[xy[0], xy[1]], terminal_type=terminal_type, src_id=src or tid)
        flag("dead_end" if terminal_type in DEAD else "open_end", n, f"trace {tid} stops at {terminal_type}")
        return n

    # endpoints: node id, or a tee request resolved below
    reqs = []  # dict(tr, end, xy)
    for tr in traces:
        t = tr.raw
        # start
        if t["source_obj_type"] in CONNECTORS:
            tr.cuts.append([0.0, conn_node[t["source_obj_id"]], tr.pts[0]])
        elif t.get("trace_kind") == "port" and t["source_obj_id"] in eq_tag:
            n = resolve_port(t["source_obj_id"], t.get("port_index"), (t["port"]["x"], t["port"]["y"]))
            tr.cuts.append([0.0, n or eq_tag[t["source_obj_id"]], tr.pts[0]])
            if not n:
                flag("port_unresolved", tr.id, "start port not found; attached to equipment node")
        else:  # branch: starts on another trace
            reqs.append({"tr": tr, "end": "s", "xy": tr.pts[0]})
        # end
        tt, tid, txy = t["terminal_type"], t.get("terminal_obj_id"), tuple(t["terminal_xy"])
        if tt == "tee_junction":
            reqs.append({"tr": tr, "end": "e", "xy": txy})
        elif tt == "equipment" and tid in eq_tag:
            ports = [p for p in t.get("attachments", {}).get("equipment_ports", []) if p.get("source_obj_id") == tid]
            n = None
            if ports:
                p = max(ports, key=lambda q: q.get("t", 0))
                n = resolve_port(tid, p.get("port_index"), tuple(p["port_xy"]))
            n = n or resolve_port(tid, None, txy)
            if not n:
                flag("port_unresolved", tr.id, "arrival port not found; attached to equipment node")
            tr.cuts.append([tr.L, n or eq_tag[tid], txy])
        elif tid in conn_node:
            tr.cuts.append([tr.L, conn_node[tid], txy])
        else:
            tr.cuts.append([tr.L, end_node(txy, tr.id, tt, tid), txy])

    # ---- tee resolution -------------------------------------------------------------------
    for r in reqs:
        best = None
        for o in traces:
            if o is r["tr"]:
                continue
            d, s, p = o.project(r["xy"])
            if d <= TEE_REACH and (best is None or d < best[0]):
                best = (d, o, s, p)
        r["hit"] = best
    clusters = []  # dict(anchor, reqs, splits{trace.id:(trace,s,p)}, node)
    for r in reqs:
        pts = [r["xy"]] + ([r["hit"][3]] if r["hit"] else [])
        for c in clusters:
            if any(math.dist(p, c["anchor"]) <= TEE_MERGE for p in pts):
                break
        else:
            c = {"anchor": r["hit"][3] if r["hit"] else r["xy"], "reqs": [], "splits": {}, "node": None}
            clusters.append(c)
        c["reqs"].append(r)
        r["cluster"] = c
    clusters.sort(key=lambda c: (c["anchor"][1], c["anchor"][0]))
    tee_n = 0
    for c in clusters:
        for r in c["reqs"]:
            if not r["hit"]:
                continue
            _, o, s, p = r["hit"]
            if s <= TEE_MERGE or s >= o.L - TEE_MERGE:
                # lands on the target's own end: that end's node is the junction
                ends = [x for x in o.cuts if abs(x[0] - (0 if s <= TEE_MERGE else o.L)) < 1e-6]
                if ends and c["node"] is None:
                    c["node"] = ends[0][1]
                continue
            c["splits"].setdefault(o.id, (o, s, p))
        if c["node"] is None:
            if len(c["reqs"]) + len(c["splits"]) < 2:
                r = c["reqs"][0]
                tid = r["tr"].id
                flag("unresolved_tee", tid, f"tee at {tuple(round(v) for v in r['xy'])} found no other trace within {TEE_REACH:.0f}px")
                c["node"] = end_node(r["xy"], tid, "tee_junction")
            else:
                tee_n += 1
                c["node"] = add_node(f"{sh}/T{tee_n}", "tee", xy_px=[round(c["anchor"][0], 1), round(c["anchor"][1], 1)])
        for r in c["reqs"]:
            r["tr"].cuts.append([0.0 if r["end"] == "s" else r["tr"].L, c["node"], c["anchor"]])
        for o, s, p in c["splits"].values():
            o.cuts.append([s, c["node"], c["anchor"]])

    # ---- pipe edges -----------------------------------------------------------------------
    n_e = 0
    for tr in traces:
        cuts = sorted(tr.cuts, key=lambda x: x[0])
        att = tr.raw.get("attachments", {})
        lns = [(tr.project(tuple(a["projected_xy"]))[1], a) for a in att.get("line_numbers", [])
               if a.get("review_state", "accepted") == "accepted" and (a.get("text") or "").strip()]
        inl = [(tr.project(tuple(a["projected_xy"]))[1], a) for a in att.get("inline_objects", [])]
        lns = [(s, " ".join(a["text"].split())) for s, a in lns]
        all_texts = [t for _, t in lns]
        warned = False
        flow = tr.raw.get("flow_direction_state")
        flow = flow if flow in ("forward", "reverse") else "unknown"
        for k, ((s0, u, xu), (s1, v, xv)) in enumerate(zip(cuts, cuts[1:])):
            if s1 - s0 < 1 and u == v:
                continue
            last = k == len(cuts) - 2

            def inside(s):  # ponytail: nearest-point projection decides the half; a label on a tee exactly goes right
                return s0 <= s < s1 or (last and s == s1)
            # a trace is one walk of one pipe: every piece inherits the trace's number(s)
            mine = [t for s, t in lns if inside(s)]
            # ponytail: with several different numbers a piece with none of its own gets the trace majority
            text = Counter(mine or all_texts).most_common(1)[0][0] if all_texts else ""
            if len(set(all_texts)) > 1 and not warned:
                warned = True
                flag("multiple_line_numbers", f"{sh}/E{n_e + 1}", f"trace {tr.id}: {sorted(set(all_texts))}")
            poly = tr.piece(s0, s1, xu, xv)
            n_e += 1
            edges.append({
                "id": f"{sh}/E{n_e}", "kind": "pipe", "u": u, "v": v, "line_number": text,
                "line_core": line_core(text),
                "inline_objects": sorted({f"{a['class_name']}#{a['id']}" for s, a in inl if inside(s)}),
                "flow": flow, "trace_ids": [tr.id], "length_px": round(_arc(poly), 1), "polyline": poly,
                "evidence": {"crosscheck": "none", "review": "none"}})
    for e in edges:  # invariant: an unoriented join shows up as a long straight jump
        if e["kind"] == "pipe":
            for a, b in zip(e["polyline"], e["polyline"][1:]):
                if math.dist(a, b) > MAX_JUMP:
                    raise ValueError(f"edge {e['id']} polyline jumps {math.dist(a, b):.0f}px between {a} and {b}")
            if not e["line_core"]:
                flag("empty_line_core", e["id"], f"pipe {e['u']} -> {e['v']} (trace {e['trace_ids'][0]}) has no line core")

    pipes = [e for e in edges if e["kind"] == "pipe"]
    # ---- line list ------------------------------------------------------------------------
    by_core = defaultdict(list)
    for e in pipes:
        if e["line_core"]:
            by_core[e["line_core"]].append(e)
    line_list = []
    for core, es in sorted(by_core.items()):
        text = Counter(e["line_number"] for e in es).most_common(1)[0][0]
        pl = parse_line_number(text)
        deg = Counter(n for e in es for n in (e["u"], e["v"]))
        cand = [n for n in deg if nodes[n]["kind"] not in ("tee", "end")]
        ext = sorted(n for n in cand if deg[n] == 1) or sorted(cand)
        # ponytail: leaves of the line's own subgraph; lines that loop or branch fall back to sorted first/last
        line_list.append({"line_number": text, "line_core": core, "size": pl.get("size", ""),
                          "service": pl.get("service", ""), "spec": pl.get("spec", ""),
                          "from": ext[0] if ext else "", "to": ext[-1] if len(ext) > 1 else "",
                          "edges": [e["id"] for e in es]})

    # ---- equipment connectivity + port conflicts -------------------------------------------
    conn, port_cores = [], defaultdict(set)
    for e in pipes:
        for me, other in ((e["u"], e["v"]), (e["v"], e["u"])):
            if nodes[me]["kind"] == "port":
                conn.append({"equipment": nodes[nodes[me]["equipment"]]["tag"], "port": nodes[me]["mark"],
                             "line_number": e["line_number"], "other_end": other})
                if e["line_core"]:
                    port_cores[me].add(e["line_core"])
    for p, cs in port_cores.items():
        if len(cs) > 1:
            flag("port_core_conflict", p, f"port touches edges with different line cores {sorted(cs)}")

    return {"schema": SCHEMA, "sheets": [sh], "nodes": list(nodes.values()), "edges": edges,
            "line_list": line_list, "equipment_connectivity": conn, "flags": flags}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--indir", default="garnet/tests/input")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        g = build_graph(a.stem, a.dir, a.indir)
    except FileNotFoundError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1
    out = Path(a.out) if a.out else Path(a.dir) / "graph.json"
    out.write_text(json.dumps(g, indent=1))
    kinds = lambda xs, k="kind": dict(sorted(Counter(x[k] for x in xs).items()))  # noqa: E731
    pipes = [e for e in g["edges"] if e["kind"] == "pipe"]
    print(f"{a.stem}: nodes {kinds(g['nodes'])} | edges {kinds(g['edges'])} "
          f"({sum(1 for e in pipes if e['line_core'])}/{len(pipes)} pipes with line_core) | flags {kinds(g['flags'])} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
