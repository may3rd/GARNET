"""Frozen shared types and helpers for Contract D (graph) and Contract E (golden).

Edge kinds `has_port` and `cross_sheet_continues` match garnet/knowledge_graph.py RELATIONSHIP_TYPES.
"""
import re
from typing import List, Optional, TypedDict

SCHEMA = "garnet_graph_v1"

# Node kind: equipment, port, page_connection, utility_connection, connection (OSBL battery limit),
#   sampling_point, pressure_relief_valve, tee, end (dead end the tracer stopped at)
# Edge kind: pipe, has_port, cross_sheet_continues


class Node(TypedDict, total=False):
    id: str
    kind: str
    sheet: str
    tag: str          # equipment
    type: str         # equipment
    bbox_px: List[float]   # equipment
    equipment: str    # port: owning equipment node id
    mark: str         # port: nozzle mark, e.g. "N2"
    text: str         # page_connection / utility_connection / connection
    direction: str    # page_connection: "to" | "from" | ""
    xy_px: List[float]


class Evidence(TypedDict):
    crosscheck: str   # agree | disagree | only_primary | none
    review: str       # confirmed | rejected | none


class Edge(TypedDict, total=False):
    id: str
    kind: str
    u: str
    v: str
    line_number: str
    line_core: str
    inline_objects: List[str]
    flow: str         # forward | reverse | unknown
    trace_ids: List[str]
    length_px: float
    polyline: List[List[float]]
    evidence: Evidence


# "from"/"to" are Python keywords-safe as dict keys but not as class attrs, so functional form.
LineListRow = TypedDict("LineListRow", {
    "line_number": str, "line_core": str, "size": str, "service": str, "spec": str,
    "from": str, "to": str, "edges": List[str]})


class EquipmentConnection(TypedDict):
    equipment: str
    port: str
    line_number: str
    other_end: str


class Flag(TypedDict):
    kind: str
    where: str
    why: str


class Graph(TypedDict):
    schema: str
    sheets: List[str]
    nodes: List[Node]
    edges: List[Edge]
    line_list: List[LineListRow]
    equipment_connectivity: List[EquipmentConnection]
    flags: List[Flag]


class _GoldenConnectionReq(TypedDict):
    a: str
    b: str
    line_core: str


class GoldenConnection(_GoldenConnectionReq, total=False):
    """Directed PROCESS-FLOW path: `a` is upstream, `b` downstream; fluid can flow
    from a to b along the pipe. Pipe joining the two ends is not enough: a pair joined
    by pipe but with no flow path a -> b belongs in Golden.not_connected.
    Optional keys: path_px, confidence, why, evidence, verdict_note (engineer's note)."""
    path_px: List[List[float]]  # waypoints along the real pipe a -> b, sheet pixels
    confidence: str
    why: str
    evidence: str
    verdict_note: str       # engineer review note, carried from the review verdict


class GoldenInline(TypedDict):
    line_core: str
    # key "class" is a keyword; use dict access g["class"]
    count: int


class GoldenNotConnected(TypedDict):
    a: str
    b: str
    why: str


class Golden(TypedDict):
    """Answer key. `connections` are directed flow paths (a upstream -> b downstream).
    `not_connected` lists pairs with NO flow path a -> b, including pairs whose ends are
    joined by pipe but where flow cannot go from a to b (e.g. pump suction -> discharge)."""
    sheet: str
    verified_by: str
    verified_on: str  # YYYY-MM-DD
    connections: List[GoldenConnection]
    line_numbers: List[str]
    inline: List[dict]          # {"line_core", "class", "count"}
    cross_sheet: List[GoldenConnection]  # ids carry short sheet prefix
    not_connected: List[GoldenNotConnected]


class LnAssign(TypedDict):
    id: str                  # Stage 6 line-number id, e.g. "line_number_0005"
    choice: str              # candidate letter "A"|"B"|... or "none" (label belongs to no traced pipe)
    trace_id: Optional[str]  # chosen candidate's trace id, None for "none"
    xy: Optional[List[float]]  # chosen candidate's point, None for "none"
    via: str                 # alongside | leader_arrow | other
    why: str                 # one line


class LnAssignment(TypedDict):
    """Vision-agent output for line-number assignment: {"sheet": stem, "assignments": [LnAssign]}.

    The agent writes it to out/<stem>/ln_assign.json; build_graph applies it over Stage 6's geometric pick.
    """
    sheet: str
    assignments: List[LnAssign]


_SIZE = r'(?:\d+[½¼¾⅜⅝⅞]|\d+\s+\d+/\d+|\d+/\d+|\d+|[½¼¾⅜⅝⅞])"'
_CORE = re.compile(r'(?<!\d)(\d{2})\s*-\s*(\d{6})(?!\d)')
_LINE = re.compile(
    rf'^\s*({_SIZE})\s*-\s*([A-Za-z][A-Za-z0-9]*)\s*-\s*(\d{{2}})\s*-\s*(\d{{6}})\s*-\s*([A-Za-z0-9]+)'
    r'(?:\s*-\s*([A-Za-z]+))?\s*$')


def line_core(text: str) -> str:
    """Return the NN-NNNNNN core of a line number ('' if none); the join key across page breaks."""
    m = _CORE.search(text or "")
    return f"{m[1]}-{m[2]}" if m else ""


def parse_line_number(text: str) -> dict:
    """Split size-service-unit-number-spec[-insulation] into a dict; {} if it does not parse."""
    m = _LINE.match(text or "")
    if not m:
        return {}
    size, service, unit, number, spec, ins = m.groups()
    return {"size": size, "service": service, "unit": unit, "number": number,
            "spec": spec, "insulation": ins or "", "line_core": f"{unit}-{number}"}


def short_sheet(stem: str) -> str:
    """'14780-8120-25-25-0002A' -> '0002A' (last hyphen-separated token)."""
    return stem.rsplit("-", 1)[-1]


if __name__ == "__main__":
    assert line_core('4"-CUL-25-002007-B1A2-NI') == line_core('3"-CUL-25-002007-B1A2-NI') == "25-002007"
    assert line_core('4" - CUL - 25 - 002007 - B1A2') == "25-002007"
    assert parse_line_number('1 1/2"-PL-25-007004-N2A1-ET') == {
        "size": '1 1/2"', "service": "PL", "unit": "25", "number": "007004",
        "spec": "N2A1", "insulation": "ET", "line_core": "25-007004"}
    p = parse_line_number('½"-PWD-25-002016-L1A1-NI')
    assert p["size"] == '½"' and p["service"] == "PWD" and p["line_core"] == "25-002016", p
    assert parse_line_number('3/4"-F-25-000001-B1A2-NI')["size"] == '3/4"'
    p = parse_line_number('1½"-PL-25-007004-N2A1-ET')
    assert p["size"] == '1½"' and p["line_core"] == "25-007004" and p["insulation"] == "ET", p
    assert parse_line_number('2 1/2"-PL-25-007004-N2A1-ET')["size"] == '2 1/2"'
    p = parse_line_number('2"-S12-66-059061-MS12-H')
    assert (p["service"], p["unit"], p["number"], p["spec"], p["insulation"]) == ("S12", "66", "059061", "MS12", "H"), p
    assert parse_line_number("25-0003") == {}
    assert line_core("PI 0203") == "" and parse_line_number("PI 0203") == {}
    assert short_sheet("14780-8120-25-25-0002A") == "0002A"
    print("OK")
