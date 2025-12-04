import re
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


def set_page_style() -> None:
    """Configure Streamlit page and inject glassmorphism CSS."""
    st.set_page_config(
        page_title="AI-Powered Code Documentation Generator",
        layout="wide",
        page_icon="📘",
    )

    glass_css = """
    <style>
    body {
        background: radial-gradient(circle at top left, #1e0533 0, #050512 45%, #020617 100%) !important;
    }
    .stApp {
        background: radial-gradient(circle at top left, #1e0533 0, #050512 45%, #020617 100%) !important;
        color: #f9fafb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }
    /* Center main heading */
    .main-header {
        text-align: center;
        padding-top: 1.5rem;
        padding-bottom: 0.5rem;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow:
            0 0 40px rgba(168, 85, 247, 0.4),
            0 0 80px rgba(56, 189, 248, 0.15);
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
    }
    .glass-card-muted {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .neon-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(168, 85, 247, 0.8);
        background: radial-gradient(circle at top left, rgba(129, 140, 248, 0.36), rgba(15, 23, 42, 0.9));
        color: #e5e7eb;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .neon-pill span.icon {
        font-size: 0.95rem;
    }
    /* Buttons */
    .stButton>button {
        border-radius: 999px;
        border: 1px solid rgba(168, 85, 247, 0.85);
        background: radial-gradient(circle at top left, #a855f7, #6366f1);
        color: #f9fafb;
        padding: 0.5rem 1.4rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        box-shadow:
            0 10px 30px rgba(56, 189, 248, 0.35),
            0 0 25px rgba(147, 51, 234, 0.7);
        transition: all 0.18s ease-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow:
            0 18px 45px rgba(56, 189, 248, 0.55),
            0 0 35px rgba(147, 51, 234, 0.9);
    }
    .stButton>button:focus {
        outline: none;
        border-color: rgba(248, 250, 252, 0.9);
        box-shadow:
            0 0 0 1px rgba(248, 250, 252, 0.65),
            0 0 30px rgba(129, 140, 248, 0.8);
    }
    /* Inputs */
    .stTextInput>div>div>input,
    .stTextArea textarea,
    .stSelectbox>div>div>select {
        background: rgba(15, 23, 42, 0.85);
        color: #e5e7eb;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.65);
    }
    .stTextArea textarea {
        border-radius: 18px;
    }
    .stSidebar {
        background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.9));
        border-right: 1px solid rgba(148, 163, 184, 0.25);
    }
    .sidebar-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9ca3af;
        margin-bottom: 0.75rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.16em;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    .logs-header {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #9ca3af;
        margin-bottom: 0.25rem;
    }
    .success-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.18);
        border: 1px solid rgba(34, 197, 94, 0.45);
        color: #bbf7d0;
        font-size: 0.8rem;
    }
    .error-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid rgba(239, 68, 68, 0.45);
        color: #fecaca;
        font-size: 0.8rem;
    }
    </style>
    """
    st.markdown(glass_css, unsafe_allow_html=True)


def clone_github_repo(repo_url: str) -> Tuple[Optional[Path], Optional[str]]:
    """Clone a GitHub repository into a temporary directory."""
    if not repo_url.strip():
        return None, "GitHub URL is empty."
    if "github.com" not in repo_url:
        return None, "Only GitHub URLs are supported."

    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="repo_clone_"))
        cmd = ["git", "clone", "--depth", "1", repo_url.strip(), str(tmp_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None, f"Failed to clone repository: {result.stderr.strip()}"
        return tmp_dir, None
    except FileNotFoundError:
        return None, "Git is not installed or not available on PATH."
    except Exception as exc:
        return None, f"Unexpected error while cloning repository: {exc}"


def run_pipeline(project_path: Path, output_dir: Path, log_container) -> Tuple[int, str]:
    """
    Run the automate_docs pipeline and stream logs to the UI.

    Returns (exit_code, collected_logs).
    """
    # Use the same Python interpreter that is running Streamlit
    python_exec = sys.executable or "python"
    project_root = Path(__file__).resolve().parent

    cmd = [
        python_exec,
        "automate_docs.py",
        "--project_path",
        str(project_path),
        "--output-dir",
        str(output_dir),
        "--read_from_input",
    ]

    log_lines = []
    log_area = log_container.empty()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(project_root),
        )
    except Exception as exc:
        message = f"Failed to start pipeline process: {exc}"
        log_area.code(message, language="bash")
        return 1, message

    with log_area:
        st.markdown('<div class="logs-header">Pipeline Logs</div>', unsafe_allow_html=True)
        log_box = st.empty()

        if process.stdout is not None:
            for line in process.stdout:
                log_lines.append(line.rstrip("\n"))
                # Show only last N lines to keep UI responsive
                tail = "\n".join(log_lines[-400:])
                log_box.code(tail, language="bash")

        process.wait()

    collected = "\n".join(log_lines)
    return process.returncode, collected


def load_documentation_files(output_dir: Path, project_path: Path) -> dict:
    """
    Load all generated documentation files from the output directory.
    Parses index.md to find referenced documentation files and loads their content.
    
    Args:
        output_dir: Directory where documentation is stored
        project_path: Original project directory path
    
    Returns:
        Dictionary with file paths as keys and content as values
    """
    docs = {}
    
    # Load index.md (always at root of output_dir)
    index_path = output_dir / "index.md"
    index_content = ""
    if index_path.exists():
        try:
            index_content = index_path.read_text(encoding="utf-8")
            docs["index.md"] = index_content
        except Exception:
            docs["index.md"] = ""
    
    # Load README.md (always at root of output_dir)
    readme_path = output_dir / "README.md"
    if readme_path.exists():
        try:
            docs["README.md"] = readme_path.read_text(encoding="utf-8")
        except Exception:
            docs["README.md"] = ""
    
    # Parse index.md to find all markdown links and load those files
    # Links are in format: [filename](path/to/file.md)
    if index_content:
        # Pattern to match markdown links: [text](path)
        # This handles both forward and backslashes in paths
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, index_content)
        
        for link_text, link_path in links:
            try:
                # Path handles both forward and backslashes automatically
                doc_file_path = Path(link_path)
                
                # Check if it's an absolute path
                if doc_file_path.is_absolute() or (len(str(link_path)) > 1 and str(link_path)[1] == ':'):
                    # Absolute path (Windows: C:\ or Unix: /)
                    if doc_file_path.exists() and doc_file_path.suffix == '.md':
                        try:
                            content = doc_file_path.read_text(encoding="utf-8")
                            display_name = link_text.strip() if link_text else doc_file_path.name
                            if display_name not in docs:  # Avoid duplicates
                                docs[display_name] = content
                        except Exception:
                            pass
                else:
                    # Relative path - try output_dir first, then project_path
                    for base_path in [output_dir, project_path]:
                        rel_path = base_path / doc_file_path
                        if rel_path.exists() and rel_path.suffix == '.md':
                            try:
                                content = rel_path.read_text(encoding="utf-8")
                                display_name = link_text.strip() if link_text else rel_path.name
                                if display_name not in docs:  # Avoid duplicates
                                    docs[display_name] = content
                                break  # Found it, no need to check other locations
                            except Exception:
                                pass
            except Exception:
                # If path parsing fails, skip this link
                pass
    
    # Also load any .md files directly in output_dir structure (fallback)
    for md_file in output_dir.rglob("*.md"):
        rel_path = md_file.relative_to(output_dir)
        rel_str = str(rel_path)
        
        # Skip index.md and README.md (already loaded)
        if rel_str in ["index.md", "README.md"]:
            continue
        
        # Skip if we already loaded this file from index.md links
        if any(rel_str in str(k) or md_file.name in str(k) for k in docs.keys() if k not in ["index.md", "README.md"]):
            continue
        
        try:
            content = md_file.read_text(encoding="utf-8")
            
            # Only include files that look like generated documentation
            is_generated_doc = (
                content and 
                len(content.strip()) > 50 and 
                ("# File:" in content or "## Chunk" in content or "File Summary" in content or "Purpose:" in content)
            )
            
            if is_generated_doc:
                # Use a clean display name
                display_name = rel_str.replace('\\', '/')  # Normalize path separators
                docs[display_name] = content
        except Exception:
            pass
    
    return docs


def build_dependency_graph_html(
    nodes_csv: Path,
    edges_csv: Path,
) -> Optional[str]:
    """
    Build an interactive function dependency graph using pyvis and
    return the generated HTML as a string for embedding in Streamlit.
    """
    if not nodes_csv.exists() or not edges_csv.exists():
        return None

    G = nx.DiGraph()

    # Load nodes
    with nodes_csv.open(encoding="utf-8") as nf:
        reader = csv.DictReader(nf)
        for row in reader:
            name = (row.get("Name") or "").strip()
            if not name:
                continue
            desc = row.get("Description", "")
            G.add_node(name, title=desc)

    # Load edges
    with edges_csv.open(encoding="utf-8") as ef:
        reader = csv.DictReader(ef)
        for row in reader:
            from_name = (row.get("From Name") or "").strip()
            to_name = (row.get("To Name") or "").strip()
            edge_type = (row.get("Edge Type") or "CALLS").strip() or "CALLS"
            if not from_name or not to_name:
                continue
            to_names = [t.strip() for t in to_name.split(";") if t.strip()]
            for t in to_names:
                G.add_edge(from_name, t, label=edge_type)

    # Remove header-like nodes
    invalid_nodes = {
        "Column",
        "From",
        "To",
        "Name",
        "Node Type",
        "From Type",
        "To Type",
        "Edge Type",
    }
    to_remove = [n for n in G.nodes() if str(n).strip() in invalid_nodes]
    if to_remove:
        G.remove_nodes_from(to_remove)

    if G.number_of_nodes() == 0:
        return None

    # Create pyvis network with dark theme-aligned background
    net = Network(
        height="640px",
        width="100%",
        bgcolor="#020617",
        font_color="#e5e7eb",
        directed=True,
    )

    # Add nodes with indegree/outdegree info in the tooltip
    for n, data in G.nodes(data=True):
        indeg = G.in_degree(n)
        outdeg = G.out_degree(n)
        desc = data.get("title", "")
        title_lines = [f"<b>{n}</b>", f"In-degree: {indeg}", f"Out-degree: {outdeg}"]
        if desc:
            title_lines.append(desc)
        title_html = "<br>".join(title_lines)
        net.add_node(n, label=n, title=title_html)

    # Add edges
    for u, v, data in G.edges(data=True):
        edge_type = data.get("label", "CALLS")
        net.add_edge(u, v, label=edge_type, title=edge_type)

    # Enable simple, clean physics and smooth edges (no debug options)
    net.set_options(
        """
        var options = {
          "nodes": {
            "shape": "dot",
            "size": 18,
            "font": {
              "size": 14,
              "color": "#e5e7eb"
            }
          },
          "edges": {
            "color": {
              "color": "#9ca3af"
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.7
              }
            }
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 120
            },
            "barnesHut": {
              "gravitationalConstant": -4000,
              "centralGravity": 0.2,
              "springLength": 120,
              "springConstant": 0.04,
              "damping": 0.09
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 120
          }
        }
        """
    )

    # Generate HTML
    html = net.generate_html(notebook=False)

    # Inject a search bar + reset button above the graph and JS for
    # search, isolation, and reset behavior.
    controls_html = """
    <div id="graph-controls" style="margin-bottom:8px; display:flex; gap:8px; align-items:center; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif; font-size: 13px;">
      <input id="graph-search-input" type="text" placeholder="Search node..." 
             style="flex:1; padding:4px 10px; border-radius:999px; border:1px solid #4b5563; background-color:#020617; color:#e5e7eb; outline:none;">
      <button id="graph-reset-btn" 
              style="padding:4px 12px; border-radius:999px; border:1px solid #4b5563; background-color:#111827; color:#e5e7eb; cursor:pointer;">
        Reset
      </button>
    </div>
    """

    # Wrap the graph and controls in a styled container and place controls
    container_start = (
        '<div class="pyvis-graph-container" '
        'style="width:100%; box-sizing:border-box; padding:12px; border-radius:12px; '
        'background: rgba(15,23,42,0.6); box-shadow: 0 8px 24px rgba(2,6,23,0.6); '
        'backdrop-filter: blur(6px); margin-bottom:12px;">'
    )
    # Insert container start + controls before the main network div
    html = html.replace('<div id="mynetwork"', container_start + controls_html + '<div id="mynetwork"', 1)

    # Inject JS for click isolation, search, and reset + info panel
    # Additional CSS to ensure the PyVis canvas blends well and is responsive
    container_css = """
    <style>
    /* Ensure the pyvis container is full width and blends with Streamlit theme */
    .pyvis-graph-container { max-width: 100%; }
    .pyvis-graph-container .vis-network { border-radius: 10px !important; }
    .pyvis-graph-container .vis-network canvas { border-radius: 10px !important; }
    /* Remove any white backgrounds inside the generated html */
    html, body { background: transparent !important; }
    .pyvis-graph-container { background: rgba(15,23,42,0.6) !important; }
    /* Controls tweaks */
    #graph-controls input { box-shadow: none; }
    #graph-controls button { box-shadow: none; }
    </style>
    """

    custom_js = """
    (function () {
        var infoDiv = document.getElementById('node-info');
        if (!infoDiv) {
            infoDiv = document.createElement('div');
            infoDiv.id = 'node-info';
            infoDiv.style.position = 'absolute';
            infoDiv.style.bottom = '10px';
            infoDiv.style.right = '10px';
            infoDiv.style.background = 'rgba(15,23,42,0.95)';
            infoDiv.style.color = '#e5e7eb';
            infoDiv.style.padding = '8px 12px';
            infoDiv.style.borderRadius = '8px';
            infoDiv.style.fontSize = '12px';
            infoDiv.style.maxWidth = '280px';
            infoDiv.style.zIndex = 9999;
            infoDiv.innerHTML = 'Click a node to isolate its neighborhood. Use search to focus a node. Click Reset to show full graph.';
            document.body.appendChild(infoDiv);
        }

        var allNodes = network.body.data.nodes.get();

        // Helper: explicit default color used for resetting nodes
        var DEFAULT_NODE_COLOR = { background: '#1E90FF', border: '#0f4f7c' };
        var HIGHLIGHT_COLOR = { background: '#f97316', border: '#b45309' };

        function resetGraphView() {
            var updates = [];
            allNodes.forEach(function (n) {
                updates.push({ id: n.id, hidden: false });
            });
            network.body.data.nodes.update(updates);
            network.fit();
            infoDiv.innerHTML = 'View reset. Click a node to isolate its neighborhood.';
        }

        // Node click isolation
        network.on('click', function (params) {
            if (params.nodes.length === 0) {
                return;
            }

            var nodeId = params.nodes[0];
            var neighbors = network.getConnectedNodes(nodeId);
            var neighborhood = neighbors.concat([nodeId]);

            var updates = [];
            allNodes.forEach(function (n) {
                var hide = neighborhood.indexOf(n.id) === -1;
                updates.push({ id: n.id, hidden: hide });
            });
            network.body.data.nodes.update(updates);

            var nodeData = network.body.data.nodes.get(nodeId);
            if (nodeData && nodeData.title) {
                infoDiv.innerHTML = nodeData.title;
            } else {
                infoDiv.innerHTML = 'Node: ' + nodeId + '<br>Neighbors: ' + neighbors.join(', ');
            }
        });

        // Reset button
        var resetBtn = document.getElementById('graph-reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', function () {
                resetGraphView();
            });
        }

        // Search behavior (client-side, on the embedded search box)
        var searchInput = document.getElementById('graph-search-input');
        if (searchInput) {
            searchInput.addEventListener('keyup', function (e) {
                var query = (e.target.value || '').toLowerCase().trim();
                if (!query) {
                    // Clear search: reset visibility and explicit colors to defaults
                    var updates = [];
                    allNodes.forEach(function (n) {
                        updates.push({ id: n.id, hidden: false, color: DEFAULT_NODE_COLOR });
                    });
                    network.body.data.nodes.update(updates);
                    network.unselectAll();
                    infoDiv.innerHTML = 'Cleared search. Click a node to isolate its neighborhood.';
                    return;
                }

                var candidates = [];
                allNodes.forEach(function (n) {
                    if (String(n.label || n.id).toLowerCase().indexOf(query) !== -1) {
                        candidates.push(n.id);
                    }
                });

                if (candidates.length > 0) {
                    var targetId = candidates[0];
                    // Reset explicit colors to defaults for all nodes
                    var resetColors = [];
                    allNodes.forEach(function (n) {
                        resetColors.push({ id: n.id, color: DEFAULT_NODE_COLOR });
                    });
                    network.body.data.nodes.update(resetColors);

                    // Focus the first match (do not select, to avoid multi-select side-effects)
                    network.unselectAll();
                    network.focus(targetId, {
                        scale: 1.6,
                        animation: { duration: 500, easingFunction: 'easeInOutQuad' }
                    });

                    // Highlight only the target node explicitly
                    network.body.data.nodes.update([
                        { id: targetId, color: HIGHLIGHT_COLOR }
                    ]);

                    infoDiv.innerHTML = 'Focused on node: ' + targetId;
                } else {
                    infoDiv.innerHTML = 'No node found matching "' + query + '".';
                }
            });
        }

        // Initial fit
        network.once('stabilizationIterationsDone', function () {
            network.fit();
        });
    })();
    """

    # Inject container CSS into the <head> if possible
    if "</head>" in html:
        html = html.replace("</head>", container_css + "</head>", 1)

    if "</body>" in html:
        # Close the pyvis container div and add our custom JS
        html = html.replace(
            "</body>",
            f"</div><script type=\"text/javascript\">{custom_js}</script></body>",
        )

    return html


def render_file_tree(file_docs: dict) -> None:
    """
    Render a hierarchical folder → subfolder → file tree for documented files.
    Keys in file_docs can be either simple names or relative paths.
    """
    # Build tree structure from paths
    tree = {}
    for path_str, content in file_docs.items():
        normalized = path_str.replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = content

    def _render_subtree(subtree: dict) -> None:
        for name, value in sorted(subtree.items(), key=lambda x: x[0].lower()):
            if isinstance(value, dict):
                with st.expander(f"📁 {name}", expanded=False):
                    _render_subtree(value)
            else:
                with st.expander(f"📄 {name}", expanded=False):
                    st.markdown(value)

    _render_subtree(tree)


def validate_local_path(path_str: str) -> Tuple[Optional[Path], Optional[str]]:
    """Validate a local project directory path."""
    if not path_str.strip():
        return None, "Please provide a project directory path."
    path = Path(path_str).expanduser()
    if not path.exists():
        return None, f"Path does not exist: {path}"
    if not path.is_dir():
        return None, f"Path is not a directory: {path}"
    return path, None


def main() -> None:
    set_page_style()

    # Header
    st.markdown(
        """
        <div class="main-header">
            <div class="neon-pill"><span class="icon">*</span><span>AI-Powered Software Docs</span></div>
            <h1 style="margin-top: 0.6rem; margin-bottom: 0.25rem; font-size: 2.2rem; color: #e5e7eb;">
                Intelligent Code Documentation Generator
            </h1>
            <p style="color:#9ca3af; max-width:640px; margin: 0 auto;">
                Point the engine at any local project or GitHub repository, generate rich technical documentation,
                and view it directly in your browser.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Source",
            ["Local project directory", "GitHub repository URL"],
            index=0,
        )

        local_project_path = ""
        github_url = ""

        if input_mode == "Local project directory":
            local_project_path = st.text_input(
                "Project directory path",
                value=str(Path.cwd()),
                help="Absolute or relative path to the root of the project you want to document.",
            )
        else:
            github_url = st.text_input(
                "GitHub repository URL",
                placeholder="https://github.com/owner/repo",
            )

        output_subdir = st.text_input(
            "Output folder name (relative)",
            value="generated_docs",
            help="Documentation will be written into this folder, created relative to the working directory.",
        )

        st.markdown("---")
        run_button = st.button("Run Documentation Generator", type="primary", use_container_width=True)

    # Main content area - full width for documentation display
    main_content = st.container()

    # Initialize session state to persist last run info across reruns
    if "last_run" not in st.session_state:
        st.session_state.last_run = {
            "project_path": None,
            "output_dir": None,
            "logs": "",
            "exit_code": None,
        }

    with main_content:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        if run_button:
            # Resolve project path
            with st.spinner("Validating inputs..."):
                error_message: Optional[str] = None
                cloned_tmp: Optional[Path] = None

                if input_mode == "Local project directory":
                    project_path, error_message = validate_local_path(local_project_path)
                else:
                    cloned_tmp, error_message = clone_github_repo(github_url)
                    project_path = cloned_tmp

                if error_message:
                    st.markdown(
                        f'<div class="error-badge"><span>!</span><span>{error_message}</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Prepare output directory (relative to current working dir to avoid hardcoding)
                    output_dir = Path.cwd() / output_subdir
                    output_dir.mkdir(parents=True, exist_ok=True)

                    st.markdown(
                        f"<p style='color:#9ca3af; font-size:0.9rem;'>Running pipeline for "
                        f"<code>{project_path}</code> → <code>{output_dir}</code></p>",
                        unsafe_allow_html=True,
                    )

                    log_container = st.container()
                    with st.spinner("Running documentation pipeline... this may take a few minutes."):
                        exit_code, logs = run_pipeline(project_path, output_dir, log_container)

                    # Persist results from this run (store paths as strings for session state)
                    st.session_state.last_run = {
                        "project_path": str(project_path),
                        "output_dir": str(output_dir),
                        "logs": logs,
                        "exit_code": exit_code,
                    }

                    if exit_code == 0:
                        st.markdown(
                            '<div class="success-badge"><span>[OK]</span>'
                            '<span>Pipeline completed successfully.</span></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="error-badge"><span>[X]</span>'
                            '<span>Pipeline finished with errors. Check logs above.</span></div>',
                            unsafe_allow_html=True,
                        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Documentation Display Section
    if "last_run" in st.session_state:
        last = st.session_state.last_run
        project_path = Path(last.get("project_path")) if last.get("project_path") else None
        output_dir = Path(last.get("output_dir")) if last.get("output_dir") else None
        exit_code = last.get("exit_code")

        if output_dir and project_path and exit_code is not None and output_dir.exists():
            # Load all documentation files (pass project_path to filter correctly)
            docs = load_documentation_files(output_dir, project_path)
            
            if docs and exit_code == 0:
                st.markdown("---")
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("## Generated Documentation")
                
                # Display project info
                project_name = project_path.name
                st.markdown(
                    f"""
                    <div style="margin-bottom: 1.5rem;">
                        <div class="metric-label">Project</div>
                        <div class="metric-value">{project_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Separate index/README from file documentation
                index_content = docs.get("index.md", "")
                readme_content = docs.get("README.md", "")
                
                # Get all file documentation (excluding index.md and README.md)
                file_docs = {k: v for k, v in docs.items() if k not in ["index.md", "README.md"]}
                
                # Create tabs: Overview (index), README, and Files
                tab_names = []
                if index_content:
                    tab_names.append("Overview")
                if readme_content:
                    tab_names.append("README")
                if file_docs:
                    tab_names.append("File Documentation")
                
                if tab_names:
                    tabs = st.tabs(tab_names)
                    tab_idx = 0
                    
                    # Overview tab (index.md)
                    if index_content:
                        with tabs[tab_idx]:
                            st.markdown(index_content)
                        tab_idx += 1
                    
                    # README tab
                    if readme_content:
                        with tabs[tab_idx]:
                            st.markdown(readme_content)
                        tab_idx += 1
                    
                    # File Documentation tab - hierarchical folder → file tree
                    if file_docs:
                        with tabs[tab_idx]:
                            st.markdown("### Documented Files\n")
                            st.markdown("Browse the documentation by folder and file.\n")
                            render_file_tree(file_docs)
                        tab_idx += 1
                else:
                    st.info("No documentation files found. The pipeline may not have generated any file documentation yet.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show interactive dependency graph if CSVs are available
                nodes_csv = Path("./src/graph/graphviz/nodes_graphcommons.csv")
                edges_csv = Path("./src/graph/graphviz/connections_graphcommons.csv")

                if nodes_csv.exists() and edges_csv.exists():
                    st.markdown("---")
                    st.markdown('<div class="glass-card-muted">', unsafe_allow_html=True)
                    st.markdown("#### Function Dependency Graph")

                    graph_html = build_dependency_graph_html(
                        nodes_csv=nodes_csv,
                        edges_csv=edges_csv,
                    )
                    if graph_html is not None:
                        components.html(graph_html, height=680, scrolling=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Show file statistics
                st.markdown('<div class="glass-card-muted">', unsafe_allow_html=True)
                st.markdown("#### Generated Files")

                chunks_file = output_dir / "chunks.jsonl"
                deps_csv = output_dir / "function_dependencies.csv"

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documentation Files", len(docs))
                with col2:
                    st.metric("Chunks File", "Present" if chunks_file.exists() else "Missing")
                with col3:
                    st.metric("Dependencies CSV", "Present" if deps_csv.exists() else "Missing")

                st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


