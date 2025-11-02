"""
Quick test script to verify Neo4j connection and authentication.
"""
from dotenv import load_dotenv
import os
import logging
from neo4j import GraphDatabase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neo4j_connection():
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j settings from environment
    uri = os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')
    
    logger.info(f"Testing Neo4j connection to {uri}")
    logger.info(f"User: {user}")
    logger.info(f"Password is set: {bool(password)}")
    
    try:
        # Try to connect
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Test connection with simple query
        with driver.session() as session:
            result = session.run("RETURN 1 as n")
            record = result.single()
            if record:
                logger.info("Success! Query returned: %s", record['n'])
        
        driver.close()
        logger.info("Connection test completed successfully")
        return True
        
    except Exception as e:
        logger.error("Connection failed: %s", e)
        return False

if __name__ == '__main__':
    test_neo4j_connection()