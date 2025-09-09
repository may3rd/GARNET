import os
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List


def _uuid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def export_dexpi(graph, nodes, edges, out_path: str, doc_meta: Dict[str, Any] | None = None) -> str:
    """
    Minimal DEXPI-like XML export stub.

    This maps nodes to simple Equipment/Connector/Inline elements with Ports,
    and edges to PipingNetworkSegment elements connecting Ports by Node IDs.

    Note: This is a pragmatic starter export, not a complete ISO 15926/DEXPI model.
    """
    doc_meta = doc_meta or {}

    root = ET.Element("DEXPI")
    doc = ET.SubElement(root, "Document")
    ET.SubElement(doc, "Title").text = str(doc_meta.get("title", "P&ID Extraction"))
    ET.SubElement(doc, "SourceImage").text = str(doc_meta.get("source", ""))

    plant = ET.SubElement(root, "Plant")
    network = ET.SubElement(plant, "PipingNetwork")

    # Node elements with one default Port each
    node_port_map: Dict[int, str] = {}
    nodes_el = ET.SubElement(network, "Nodes")
    for n in nodes:
        n_el = ET.SubElement(nodes_el, "Node", attrib={
            "id": str(n.id),
            "type": getattr(n.type, "value", str(n.type)),
        })
        pos = ET.SubElement(n_el, "Position")
        pos.set("x", str(n.position[0]))
        pos.set("y", str(n.position[1]))
        if n.label:
            ET.SubElement(n_el, "Label").text = str(n.label)
        port_id = _uuid(f"port-{n.id}")
        ET.SubElement(n_el, "Port", attrib={"id": port_id})
        node_port_map[n.id] = port_id

    # Segments (edges)
    segs_el = ET.SubElement(network, "Segments")
    for e in edges:
        seg_el = ET.SubElement(segs_el, "PipingNetworkSegment", attrib={
            "id": str(e.id)
        })
        frm = ET.SubElement(seg_el, "From")
        frm.set("node", str(e.source))
        frm.set("port", node_port_map.get(e.source, ""))
        to = ET.SubElement(seg_el, "To")
        to.set("node", str(e.target))
        to.set("port", node_port_map.get(e.target, ""))

        # Attributes (length as example)
        attrs = ET.SubElement(seg_el, "Attributes")
        if e.attributes:
            for k, v in e.attributes.items():
                a = ET.SubElement(attrs, "Attribute", attrib={"name": str(k)})
                a.text = str(v)

        # Polyline path
        if e.path:
            path_el = ET.SubElement(seg_el, "Path")
            for pt in e.path:
                p = ET.SubElement(path_el, "P")
                p.set("x", str(pt[0]))
                p.set("y", str(pt[1]))

    # Serialize
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path

