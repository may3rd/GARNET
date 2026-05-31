import json

images = ["00008", "00005", "00001"]
modes = [("rect", "RECT"), ("geo", "GEO")]

header = f"{'Image':<12} {'Mode':<4} {'Edges':>6} {'Nodes':>6} {'Comps':>6} {'Anomal':>6} {'ArtPts':>6} {'N-Cross':>8} {'Gaps':>6} {'GapCov':>7}"
print(header)
print("-" * 75)

for img in images:
    for tag, label in modes:
        dir_name = f"smoke_{tag}" if img == "00008" else f"smoke_{tag}_{img[-2:]}"
        
        gs = json.load(open(f"output/{dir_name}/stage12_graph_summary.json"))
        qa = json.load(open(f"output/{dir_name}/stage13_graph_qa_summary.json"))
        cv = json.load(open(f"output/{dir_name}/stage12_connection_validation_summary.json"))
        
        edges = gs.get("edge_count", gs.get("kept_edge_count", 0))
        nodes = gs.get("node_count", 0)
        comps = gs.get("edge_component_count", 0)
        anomal = qa.get("total_anomalies", 0)
        artpts = qa.get("articulation_point_count", 0)
        ncross = gs.get("non_connecting_crossing_count", 0)
        gaps = cv.get("total_gaps_in_summary", 0)
        gapcov = cv.get("gap_coverage_pct", 0)
        
        print(f"Test-{img:<4} {label:<4} {edges:>6} {nodes:>6} {comps:>6} {anomal:>6} {artpts:>6} {ncross:>8} {gaps:>6} {gapcov:>6.1f}%")
    print()
