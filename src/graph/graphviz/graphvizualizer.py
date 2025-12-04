import sys
import csv
import networkx as nx
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QComboBox, QSpinBox, QCheckBox, QFrame
)
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure


class GraphVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.graph = nx.DiGraph()
        self.full_pos = {}        # positions for the full graph
        self.display_pos = {}     # positions for current displayed graph (full or isolated)
        self.display_graph = None
        self.selected_node = None
        self.highlighted_nodes = set()
        self.isolated = False  # track if we are in isolation mode

        self.init_ui()
        self.load_graph_data()
        self.calculate_full_layout()
        self.reset_display_to_full()
        self.update_graph()

    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Interactive Graph Visualizer (fixed)")
        self.setGeometry(100, 100, 1600, 900)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = self.create_control_panel()

        right_panel = QVBoxLayout()
        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        right_panel.addWidget(self.toolbar)
        right_panel.addWidget(self.canvas)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 4)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
                color: #EAEAEA;
            }
            QLabel {
                color: #EAEAEA;
                font-size: 12px;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #2B2B2B;
                color: #EAEAEA;
                border: 1px solid #3A3A3A;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #1496FF;
            }
            QCheckBox {
                color: #EAEAEA;
            }
        """)

        self.show()

    def create_control_panel(self):
        """Create the left control panel"""
        layout = QVBoxLayout()
        layout.setSpacing(12)

        title = QLabel("Graph Controls")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        layout.addWidget(QLabel("🔍 Search Node:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter node name...")
        self.search_input.textChanged.connect(self.on_search)
        # Pressing Enter will attempt to isolate the first matching node
        self.search_input.returnPressed.connect(self.on_search_activate)
        layout.addWidget(self.search_input)

        layout.addWidget(QLabel("🧭 Filter by Edge Type:"))
        self.edge_filter = QComboBox()
        self.edge_filter.addItem("All Types")
        self.edge_filter.currentTextChanged.connect(self.update_graph)
        layout.addWidget(self.edge_filter)

        self.physics_toggle = QCheckBox("Enable Force Layout (recompute positions)")
        self.physics_toggle.setChecked(True)
        layout.addWidget(self.physics_toggle)

        layout.addWidget(QLabel("⚙️ Node Size:"))
        self.node_size_spin = QSpinBox()
        self.node_size_spin.setRange(100, 1500)
        self.node_size_spin.setValue(400)
        self.node_size_spin.valueChanged.connect(self.update_graph)
        layout.addWidget(self.node_size_spin)

        refresh_btn = QPushButton("🔄 Refresh Layout")
        refresh_btn.clicked.connect(self.recalculate_layout)
        layout.addWidget(refresh_btn)

        reset_btn = QPushButton("🧹 Reset View")
        reset_btn.clicked.connect(self.reset_view)
        layout.addWidget(reset_btn)

        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)

        layout.addStretch()

        panel_frame = QFrame()
        panel_frame.setLayout(layout)
        panel_frame.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border-right: 2px solid #2C2C2C;
                padding: 15px;
            }
        """)

        container = QVBoxLayout()
        container.setContentsMargins(0, 0, 0, 0)
        container.addWidget(panel_frame)
        return container

    def load_graph_data(self):
        """Load graph from CSV files and clean header-like nodes"""
        nodes_path = "./src/graph/graphviz/nodes_graphcommons.csv"
        edges_path = "./src/graph/graphviz/connections_graphcommons.csv"

        try:
            with open(nodes_path, encoding='utf-8') as nf:
                reader = csv.DictReader(nf)
                for row in reader:
                    name = row.get("Name", "").strip()
                    if not name:
                        continue
                    desc = row.get("Description", "")
                    self.graph.add_node(name, title=desc)

            edge_types = set()
            with open(edges_path, encoding='utf-8') as ef:
                reader = csv.DictReader(ef)
                for row in reader:
                    from_name = row.get("From Name", "").strip()
                    to_name = row.get("To Name", "").strip()
                    edge_type = row.get("Edge Type", "CALLS").strip() or "CALLS"
                    if not from_name or not to_name:
                        continue
                    # If To Name contains semicolons (multiple targets), split
                    to_names = [t.strip() for t in to_name.split(";") if t.strip()]
                    for t in to_names:
                        self.graph.add_edge(from_name, t, label=edge_type)
                        edge_types.add(edge_type)

            # Remove likely-bad nodes that are header words or stray tokens
            invalid_nodes = {"Column", "From", "To", "Name", "Node Type", "From Type", "To Type", "Edge Type"}
            to_remove = [n for n in self.graph.nodes() if str(n).strip() in invalid_nodes]
            if to_remove:
                self.graph.remove_nodes_from(to_remove)

            # populate edge filter
            for et in sorted(edge_types):
                if self.edge_filter.findText(et) == -1:
                    self.edge_filter.addItem(et)

        except FileNotFoundError:
            # fallback demo graph
            self.graph.add_edges_from([
                ("A", "C", {"label": "calls"}),
                ("A", "D", {"label": "calls"}),
                ("D", "A", {"label": "calls"}),
                ("B", "A", {"label": "calls"})
            ])
            # ensure nodes exist
            for n in ["A", "B", "C", "D"]:
                if n not in self.graph:
                    self.graph.add_node(n, title="demo")

        self.update_stats()

    def calculate_full_layout(self):
        """Compute layout for full graph; store in self.full_pos"""
        # If graph small, fewer iterations; for large graphs, more iterations
        iters = 120 if self.graph.number_of_nodes() < 200 else 60
        try:
            self.full_pos = nx.spring_layout(self.graph, k=1.2, iterations=iters, seed=42)
        except Exception:
            # fallback to circular layout
            self.full_pos = nx.circular_layout(self.graph)

    def reset_display_to_full(self):
        """Set display_graph and display_pos to full graph's view"""
        self.display_graph = self.graph.copy()
        self.display_pos = dict(self.full_pos)
        # Ensure display_pos has positions for all nodes in display_graph
        for n in self.display_graph.nodes():
            if n not in self.display_pos:
                self.display_pos[n] = (0.0, 0.0)

    def update_stats(self):
        """Update sidebar stats"""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
        self.stats_label.setText(f"Nodes: {num_nodes}\nEdges: {num_edges}\nAvg Degree: {avg_degree:.2f}")

    def on_search(self):
        """Highlight nodes that match search"""
        query = self.search_input.text().strip()
        self.highlighted_nodes = self.search_nodes(query)
        self.update_graph()

    def search_nodes(self, query: str):
        """Return a set of nodes matching the query (case-insensitive substring).

        This helper is safe for non-string node keys by coercing to `str` before
        lowercasing. It returns the actual node objects (not their string forms)
        which keeps compatibility with NetworkX node keys.
        """
        if not query:
            return set()
        q = query.strip().lower()
        matches = set()
        for n in self.graph.nodes():
            try:
                name = str(n)
            except Exception:
                continue
            if q in name.lower():
                matches.add(n)
        return matches

    def on_search_activate(self):
        """Called when the user presses Enter in the search box.

        If there is at least one matching node, isolate the first match so the
        user can focus on it immediately.
        """
        matches = list(self.search_nodes(self.search_input.text()))
        if not matches:
            return
        first = matches[0]
        # If the node exists in the current graph, isolate it
        if first in self.graph:
            self.isolate_node(first)
            # ensure the highlight set includes the isolated node
            self.highlighted_nodes = {first}
            self.update_graph(isolated=True)

    def on_click(self, event):
        """Handle mouse clicks on canvas to detect node selection"""
        if event.xdata is None or event.ydata is None:
            return

        # Use display_pos (positions for whatever is currently shown)
        min_dist = float('inf')
        closest_node = None
        # positions are in display_pos
        for node, (x, y) in self.display_pos.items():
            # distance in layout coordinate space
            d = (x - event.xdata) ** 2 + (y - event.ydata) ** 2
            if d < min_dist:
                min_dist = d
                closest_node = node

        # threshold: adjust depending on layout bounding box
        # compute average spacing to set a reasonable threshold
        xs = [p[0] for p in self.display_pos.values()] or [0]
        ys = [p[1] for p in self.display_pos.values()] or [0]
        x_range = max(xs) - min(xs) if max(xs) != min(xs) else 1.0
        y_range = max(ys) - min(ys) if max(ys) != min(ys) else 1.0
        thresh = ((x_range + y_range) / 2.0) * 0.02  # tuned constant

        if closest_node and min_dist < thresh:
            # toggle isolation
            if self.selected_node == closest_node and self.isolated:
                self.reset_view()
            else:
                self.selected_node = closest_node
                self.isolate_node(closest_node)

    def isolate_node(self, node):
        """Show only selected node and its direct connections (incoming + outgoing),
           and recompute layout for this small subgraph so positions exist and edges are visible."""
        self.isolated = True
        self.selected_node = node

        predecessors = list(self.graph.predecessors(node))
        successors = list(self.graph.successors(node))
        sub_nodes = set(predecessors + successors + [node])

        # Build the subgraph and compute its own layout
        self.display_graph = self.graph.subgraph(sub_nodes).copy()

        if self.physics_toggle.isChecked():
            try:
                self.display_pos = nx.spring_layout(self.display_graph, k=0.8, iterations=80, seed=42)
            except Exception:
                self.display_pos = nx.circular_layout(self.display_graph)
        else:
            # map nodes to positions from full_pos when possible
            self.display_pos = {n: self.full_pos.get(n, (0.0, 0.0)) for n in self.display_graph.nodes()}

        # ensure positions exist for all nodes
        for n in self.display_graph.nodes():
            if n not in self.display_pos:
                self.display_pos[n] = (0.0, 0.0)

        self.update_graph(isolated=True)

    def update_graph(self, isolated=False):
        """Redraw graph. This uses self.display_graph and self.display_pos."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1E1E1E")
        ax.axis("off")

        graph = self.display_graph if self.display_graph is not None else self.graph
        pos = self.display_pos if self.display_pos else self.full_pos

        # Apply edge-type filter without mutating original graph
        etype = self.edge_filter.currentText()
        if etype != "All Types":
            edges_to_draw = [(u, v) for u, v, d in graph.edges(data=True) if d.get("label") == etype]
            subg = nx.DiGraph()
            subg.add_nodes_from(graph.nodes(data=True))
            subg.add_edges_from([(u, v, graph.edges[u, v]) for (u, v) in edges_to_draw])
            graph_to_draw = subg
        else:
            graph_to_draw = graph

        # Colors: selected, highlighted, normal
        node_colors = []
        for n in graph_to_draw.nodes():
            if n == self.selected_node:
                node_colors.append("#FF5555")
            elif n in self.highlighted_nodes:
                node_colors.append("#FFD700")
            else:
                node_colors.append("#1E90FF")

        node_size = self.node_size_spin.value()

        # draw nodes/labels/edges
        try:
            nx.draw_networkx_nodes(graph_to_draw, pos, node_color=node_colors, node_size=node_size, ax=ax)
            nx.draw_networkx_labels(graph_to_draw, pos, font_color="white", font_size=9, ax=ax)
            nx.draw_networkx_edges(graph_to_draw, pos, edge_color="#AAAAAA", arrows=True, arrowsize=15, ax=ax)
        except nx.NetworkXError as e:
            # If a node is missing a position, fallback: recompute a layout for the graph_to_draw
            try:
                pos2 = nx.spring_layout(graph_to_draw, k=0.8, iterations=60, seed=42)
                self.display_pos = pos2
                nx.draw_networkx_nodes(graph_to_draw, pos2, node_color=node_colors, node_size=node_size, ax=ax)
                nx.draw_networkx_labels(graph_to_draw, pos2, font_color="white", font_size=9, ax=ax)
                nx.draw_networkx_edges(graph_to_draw, pos2, edge_color="#AAAAAA", arrows=True, arrowsize=15, ax=ax)
            except Exception:
                # give up gracefully
                ax.text(0.5, 0.5, "Unable to draw graph", color="white", ha="center", va="center")

        ax.set_title(
            f"Graph View - {'Isolated: ' + self.selected_node if isolated else 'Full Graph'}",
            fontsize=12, color="white", pad=20
        )

        self.figure.tight_layout()
        self.canvas.draw()

    def recalculate_layout(self):
        """Recalculate node positions for full graph (and update display accordingly)"""
        self.calculate_full_layout()
        if not self.isolated:
            self.reset_display_to_full()
        else:
            # if currently isolated, recompute isolated layout too
            if self.selected_node:
                self.isolate_node(self.selected_node)
        self.update_graph()

    def reset_view(self):
        """Reset all filters and isolation"""
        self.selected_node = None
        self.highlighted_nodes.clear()
        self.isolated = False
        self.search_input.clear()
        self.edge_filter.setCurrentIndex(0)
        self.reset_display_to_full()
        self.update_graph()


def main():
    app = QApplication(sys.argv)
    visualizer = GraphVisualizer()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
