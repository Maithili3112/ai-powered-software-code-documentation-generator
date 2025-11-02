"""
Test script for call graph generation and Neo4j integration.
"""
import sys
from pathlib import Path
# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
import logging
import os
from neo4j import GraphDatabase
from src.graph.generate_call_graph import generate_graph_for_project

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_call_graph():
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j connection details
    neo4j_uri = os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    # Test project path (use the small test project)
    test_project = Path(__file__).parent.parent / 'test_project'
    
    logger.info("=" * 80)
    logger.info("Starting call graph test")
    logger.info("=" * 80)
    
    # Step 1: Generate and store call graph
    logger.info("\nStep 1: Generating call graph...")
    generate_graph_for_project(str(test_project), neo4j_uri=neo4j_uri)
    
    # Step 2: Query the graph from Neo4j to verify storage
    logger.info("\nStep 2: Querying stored call graph...")
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # Query 1: Get all functions
            logger.info("\nAll functions in the graph:")
            result = session.run("""
                MATCH (f:Function)
                RETURN f.name as name, f.filepath as file, 
                       f.start_line as start, f.end_line as end
                ORDER BY f.filepath, f.start_line
            """)
            
            for record in result:
                logger.info(f"Function: {record['name']}")
                logger.info(f"  File: {record['file']}")
                logger.info(f"  Lines: {record['start']}-{record['end']}")
            
            # Query 2: Get call relationships
            logger.info("\nFunction call relationships:")
            result = session.run("""
                MATCH (caller:Function)-[r:CALLS]->(callee:Function)
                RETURN caller.name as from, callee.name as to
            """)
            
            for record in result:
                logger.info(f"{record['from']} -> {record['to']}")
        
        driver.close()
        logger.info("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Error querying Neo4j: {e}")
        return False
    
    return True

if __name__ == '__main__':
    test_call_graph()