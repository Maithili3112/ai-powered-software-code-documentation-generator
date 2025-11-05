import csv
from pathlib import Path

input_file = Path("./generated_docs/function_dependencies.csv")
edges_file = Path("./src/graph/graphviz/connections_graphcommons.csv")
nodes_file = Path("./src/graph/graphviz/nodes_graphcommons.csv")

if not input_file.exists():
    raise FileNotFoundError(f"Input file not found: {input_file.resolve()}")

edges = []
nodes = set()

with open(input_file, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        from_name = row["chunk_name"].strip()
        to_name = row["dep_name"].strip()
        if not from_name or not to_name:
            continue
        edges.append({
            "From Type": "Function",
            "From Name": from_name,
            "Edge Type": "CALLS",
            "To Type": "Function",
            "To Name": to_name
        })
        nodes.add(from_name)
        nodes.add(to_name)

# Write edges file
with open(edges_file, "w", newline='', encoding="utf-8") as out_edges:
    writer = csv.DictWriter(out_edges, fieldnames=["From Type", "From Name", "Edge Type", "To Type", "To Name"])
    writer.writeheader()
    writer.writerows(edges)

# Write nodes file
with open(nodes_file, "w", newline='', encoding="utf-8") as out_nodes:
    writer = csv.DictWriter(out_nodes, fieldnames=["Node Type", "Name", "Description"])
    writer.writeheader()
    for n in sorted(nodes):
        writer.writerow({"Node Type": "Function", "Name": n, "Description": f"Function {n}"})

print(f"✅ Done! Edges: {len(edges)}, Nodes: {len(nodes)}")
print(f"Edges file: {edges_file.resolve()}")
print(f"Nodes file: {nodes_file.resolve()}")
