"""
Generate call graphs from Python code using PyCG and store in Neo4j.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import tempfile
import subprocess
import sys
import ast
from neo4j import GraphDatabase
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jGraphBuilder:
    """Builds and manages call graphs in Neo4j."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: Optional[str] = None):
        self.uri = uri
        self.user = user
        self.password = password or os.getenv("NEO4J_PASSWORD", "test-password")
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database."""
        import time
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1 as n")
                logger.info(f"Connected to Neo4j at {self.uri}")
                return
            except Exception as e:
                if "AuthenticationRateLimit" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"Neo4j rate limit hit, waiting {retry_delay}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(retry_delay)
                        continue
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def create_indexes(self):
        """Create useful indexes in Neo4j."""
        with self.driver.session() as session:
            queries = [
                "CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)",
                "CREATE INDEX function_id IF NOT EXISTS FOR (f:Function) ON (f.id)",
                "CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path)",
                "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)"
            ]
            
            for query in queries:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")
    
    def add_function(self, func_id: str, name: str, filepath: str, 
                     start_line: int, end_line: int):
        """Add a function node to the graph."""
        with self.driver.session() as session:
            query = """
            MERGE (f:Function {id: $func_id})
            SET f.name = $name,
                f.filepath = $filepath,
                f.start_line = $start_line,
                f.end_line = $end_line
            """
            session.run(query, 
                       func_id=func_id,
                       name=name,
                       filepath=filepath,
                       start_line=start_line,
                       end_line=end_line)
    
    def add_file(self, filepath: str):
        """Add a file node to the graph."""
        with self.driver.session() as session:
            query = """
            MERGE (f:File {path: $filepath})
            """
            session.run(query, filepath=filepath)
    
    def add_call_relation(self, caller_id: str, callee_id: str):
        """Add a call relationship between functions."""
        with self.driver.session() as session:
            query = """
            MATCH (a:Function {id: $caller_id})
            MATCH (b:Function {id: $callee_id})
            MERGE (a)-[r:CALLS]->(b)
            """
            session.run(query, caller_id=caller_id, callee_id=callee_id)
    
    def add_contains_relation(self, container_id: str, contained_id: str):
        """Add a CONTAINS relationship."""
        with self.driver.session() as session:
            query = """
            MATCH (a {id: $container_id})
            MATCH (b {id: $contained_id})
            MERGE (a)-[:CONTAINS]->(b)
            """
            session.run(query, container_id=container_id, contained_id=contained_id)


class PyCGCallGraphGenerator:
    """Generate call graphs using PyCG."""
    
    def __init__(self):
        self.graph_builder = None
    
    def generate_call_graph(self, root_path: str, neo4j_uri: str = "bolt://localhost:7687") -> Dict[str, Any]:
        """
        Generate call graph from Python code using PyCG.
        
        Args:
            root_path: Root directory of Python project
            neo4j_uri: Neo4j connection URI
        
        Returns:
            Dictionary containing call graph data
        """
        logger.info(f"Generating call graph for {root_path}")
        
        # Try direct connection first (like test script)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Get fresh credentials from environment
            neo4j_uri = os.getenv('NEO4J_URI', neo4j_uri)
            neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
            neo4j_password = os.getenv('NEO4J_PASSWORD')
            
            # Test connection before creating builder
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            with driver.session() as session:
                session.run("RETURN 1 as n")
            driver.close()
            
            # Now create builder with verified credentials
            self.graph_builder = Neo4jGraphBuilder(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password
            )
            self.graph_builder.create_indexes()
        except Exception as e:
            logger.error(f"Neo4j connection failed (falling back to JSON): {e}")
            self.graph_builder = None  # Force JSON fallback
        
        try:
            # Try PyCG first
            cg_data = self._run_pycg(root_path)
            if cg_data:
                self._store_in_neo4j(cg_data, root_path)
                return cg_data
        except Exception as e:
            logger.warning(f"PyCG failed: {e}")
        
        # Fallback to AST-based analysis
        logger.info("Falling back to AST-based call graph generation")
        cg_data = self._generate_ast_graph(root_path)
        self._store_in_neo4j(cg_data, root_path)
        
        return cg_data
    
    def _run_pycg(self, root_path: str) -> Optional[Dict[str, Any]]:
        """Run PyCG using subprocess for better isolation."""
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                output_file = tmp_file.name
            
            # Try to run PyCG as a subprocess by writing a small helper script.
            # This avoids complex quoting/escaping issues on Windows when using -c
            script_content = f"""
import sys
try:
    from pycg import callgraph
except Exception as e:
    print('PyCG import error:', e)
    sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: pycg_runner.py <root_path> <output_file>')
        sys.exit(1)
    rp = sys.argv[1]
    of = sys.argv[2]
    try:
        callgraph.generate_callgraph(rp, of, max_iter=5)
        print('PyCG completed successfully')
    except Exception as e:
        print('PyCG error:', e)
        sys.exit(1)
"""

            # Write helper script to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as runner:
                runner.write(script_content)
                runner_path = runner.name

            cmd = [sys.executable, runner_path, root_path, output_file]
            logger.info("Running PyCG via helper script...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and Path(output_file).exists():
                with open(output_file, 'r') as f:
                    cg_data = json.load(f)
                logger.info(f"PyCG generated call graph with {len(cg_data)} functions")
                return cg_data
            else:
                logger.warning(f"PyCG subprocess failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.warning("PyCG timed out")
            return None
        except Exception as e:
            logger.warning(f"PyCG subprocess error: {e}")
            return None
        finally:
            # Clean up temp file
            try:
                if 'output_file' in locals() and Path(output_file).exists():
                    Path(output_file).unlink()
            except:
                pass
    
    def _store_in_neo4j(self, cg_data: Dict[str, Any], root_path: str):
        """Store call graph data in Neo4j."""
        if not self.graph_builder:
            logger.warning("No graph builder available; skipping Neo4j storage")
            return

        try:
            # Add nodes and relationships
            for func_name, func_info in cg_data.items():
                # Add function node
                filepath = func_info.get('filename', '')
                start_line = func_info.get('lineno', 0)
                end_line = func_info.get('end_lineno', start_line)

                self.graph_builder.add_function(
                    func_id=func_name,
                    name=func_name,
                    filepath=filepath,
                    start_line=start_line,
                    end_line=end_line
                )

                # Add file node
                self.graph_builder.add_file(filepath)

                # Add call relationships
                for called_func in func_info.get('calls', []):
                    try:
                        self.graph_builder.add_call_relation(func_name, called_func)
                    except Exception as e:
                        logger.warning(f"Failed to add call relation {func_name} -> {called_func}: {e}")

            logger.info("Stored call graph in Neo4j")

        except Exception as e:
            logger.error(f"Error while storing call graph in Neo4j: {e}")
            # Fallback: write call graph to a JSON file for inspection
            try:
                out_file = Path(root_path) / 'callgraph_fallback.json'
                import json
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(cg_data, f, indent=2)
                logger.info(f"Wrote fallback call graph JSON to {out_file}")
            except Exception as ew:
                logger.error(f"Failed to write fallback call graph JSON: {ew}")
    
    def _generate_ast_graph(self, root_path: str) -> Dict[str, Any]:
        """Generate call graph using AST analysis."""
        logger.info("Generating AST-based call graph")
        
        cg_data = {}
        root = Path(root_path)
        
        for py_file in root.rglob("*.py"):
            try:
                # Skip hidden files and common ignore patterns
                if any(part.startswith('.') for part in py_file.parts):
                    continue
                if 'venv' in py_file.parts or '__pycache__' in py_file.parts:
                    continue
                
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                import ast
                tree = ast.parse(source)
                
                # Extract functions and classes
                visitor = CallGraphVisitor(str(py_file), source)
                visitor.visit(tree)
                
                # Add to call graph data
                for func_info in visitor.functions:
                    func_name = func_info['name']
                    cg_data[func_name] = {
                        'filename': str(py_file),
                        'lineno': func_info['lineno'],
                        'end_lineno': func_info['end_lineno'],
                        'calls': func_info['calls'],
                        'type': func_info['type']
                    }
                    
                    # Add to Neo4j
                    self.graph_builder.add_function(
                        func_id=func_name,
                        name=func_info['name'].split('.')[-1],
                        filepath=str(py_file),
                        start_line=func_info['lineno'],
                        end_line=func_info['end_lineno']
                    )
                    self.graph_builder.add_file(str(py_file))
                    
                    # Add call relationships
                    for called_func in func_info['calls']:
                        self.graph_builder.add_call_relation(func_name, called_func)
            
            except Exception as e:
                logger.warning(f"Error processing {py_file}: {e}")
        
        logger.info(f"AST analysis found {len(cg_data)} functions")
        return cg_data


class CallGraphVisitor(ast.NodeVisitor):
    """AST visitor to extract function calls and definitions."""
    
    def __init__(self, filepath: str, source: str):
        self.filepath = filepath
        self.source = source
        self.functions = []
        self.current_class = None
        self.current_function = None
    
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        func_name = node.name
        if self.current_class:
            func_name = f"{self.current_class}.{func_name}"
            func_type = 'method'
        else:
            func_type = 'function'
        
        # Extract function calls
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._extract_call_name(child)
                if call_name:
                    calls.append(call_name)
        
        func_info = {
            'name': func_name,
            'lineno': node.lineno,
            'end_lineno': node.end_lineno or node.lineno,
            'calls': calls,
            'type': func_type
        }
        
        self.functions.append(func_info)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit class definition."""
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None
    
    def _extract_call_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract function name from call node."""
        # Be resilient: accept either a Call, Attribute, or Name node and extract a dotted name
        try:
            node = call_node
            # If a Call was passed, get its func
            if isinstance(node, ast.Call):
                node = node.func

            # Name: simple function
            if isinstance(node, ast.Name):
                return node.id

            # Attribute: recurse to build dotted name
            if isinstance(node, ast.Attribute):
                parts = []
                cur = node
                # Walk back through Attribute/Name nodes
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                parts.reverse()
                return ".".join(parts)

        except Exception:
            # Anything unexpected: return None
            return None

        return None


def generate_graph_for_project(project_path: str, neo4j_uri: str = "bolt://localhost:7687") -> None:
    """
    Generate and store call graph for a project.
    
    Args:
        project_path: Path to project directory
        neo4j_uri: Neo4j connection URI
    """
    generator = PyCGCallGraphGenerator()
    try:
        generator.generate_call_graph(project_path, neo4j_uri)
    finally:
        if generator.graph_builder:
            generator.graph_builder.close()
