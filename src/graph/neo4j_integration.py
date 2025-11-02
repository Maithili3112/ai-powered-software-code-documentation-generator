#!/usr/bin/env python3
"""
Neo4j integration module for call graph storage and querying.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


class Neo4jIntegration:
    """Neo4j integration for call graph storage and querying.

    Credentials should be provided via environment variables in production:
      - NEO4J_URI (default: neo4j://127.0.0.1:7687)
      - NEO4J_USER (default: neo4j)
      - NEO4J_PASSWORD
    """

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        # Load configuration from environment
        # Initialize logger early so we can use it during init
        self.logger = logging.getLogger(__name__)

        self.uri = uri or os.getenv('NEO4J_URI')
        self.user = user or os.getenv('NEO4J_USER')
        self.password = password or os.getenv('NEO4J_PASSWORD')

        # Fallback to defaults if not in environment
        if not self.uri:
            self.uri = "bolt://localhost:7687"
            self.logger.warning("NEO4J_URI not set, using default: bolt://localhost:7687")
        if not self.user:
            self.user = "neo4j"
            self.logger.warning("NEO4J_USER not set, using default: neo4j")
        if not self.password:
            # Don't raise here — allow the integration to use a default for local/dev use
            self.logger.warning("NEO4J_PASSWORD not set. Using default 'test-password'. Set NEO4J_PASSWORD in environment for production.")
            self.password = "test-password"

        self.driver = None

        # Initialize Neo4j connection
        self._setup_neo4j()

    def _setup_neo4j(self):
        """Setup Neo4j connection."""
        if not GraphDatabase:
            raise ImportError("neo4j not installed. Run: pip install neo4j")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.logger.info("Neo4j connection established successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            self.logger.info("Neo4j integration will be skipped. Make sure Neo4j is running.")
            self.driver = None
    
    def ingest_call_graph(self, csv_file: str) -> bool:
        """Ingest call graph from CSV file into Neo4j."""
        if not self.driver:
            self.logger.warning("Neo4j not available, skipping call graph ingestion")
            return False
        
        try:
            with self.driver.session() as session:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                
                # Read CSV and create nodes and relationships
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        # Create function node
                        session.run("""
                            MERGE (f:Function {
                                name: $name,
                                file: $file,
                                start_line: $start_line,
                                end_line: $end_line
                            })
                        """, 
                        name=row.get('function_name', ''),
                        file=row.get('file_path', ''),
                        start_line=int(row.get('start_line', 0)),
                        end_line=int(row.get('end_line', 0))
                        )
                        
                        # Create call relationships
                        if row.get('calls'):
                            calls = row['calls'].split(',')
                            for call in calls:
                                call = call.strip()
                                if call:
                                    session.run("""
                                        MATCH (f:Function {name: $from_name})
                                        MERGE (t:Function {name: $to_name})
                                        MERGE (f)-[:CALLS]->(t)
                                    """, 
                                    from_name=row.get('function_name', ''),
                                    to_name=call
                                    )
                
                self.logger.info("Call graph ingested successfully into Neo4j")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to ingest call graph: {e}")
            return False
    
    def query_call_graph(self, function_name: str) -> List[Dict[str, Any]]:
        """Query call graph for a specific function."""
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (f:Function {name: $name})
                    OPTIONAL MATCH (f)-[:CALLS]->(called:Function)
                    OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
                    RETURN f, collect(DISTINCT called) as calls_out, collect(DISTINCT caller) as calls_in
                """, name=function_name)
                
                records = []
                for record in result:
                    records.append({
                        'function': dict(record['f']),
                        'calls_out': [dict(call) for call in record['calls_out'] if call],
                        'calls_in': [dict(call) for call in record['calls_in'] if call]
                    })
                
                return records
                
        except Exception as e:
            self.logger.error(f"Failed to query call graph: {e}")
            return []
    
    def get_all_functions(self) -> List[Dict[str, Any]]:
        """Get all functions in the call graph."""
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (f:Function) RETURN f")
                
                functions = []
                for record in result:
                    functions.append(dict(record['f']))
                
                return functions
                
        except Exception as e:
            self.logger.error(f"Failed to get all functions: {e}")
            return []
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")

