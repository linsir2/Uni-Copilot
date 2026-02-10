import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import qdrant_client
from neo4j import GraphDatabase

# Config
load_dotenv()
QDRANT_URL = "http://localhost:6333"
CHUNK_COLLECTION = "edu_matrix_chunks"    
ENTITY_COLLECTION = "edu_matrix_entities" 
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def check_qdrant():
    print("\n🔍 [Qdrant Health Check]")
    try:
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        
        # Check Chunks
        if client.collection_exists(CHUNK_COLLECTION):
            info = client.get_collection(CHUNK_COLLECTION)
            print(f"   ✅ Chunks ({CHUNK_COLLECTION}): {info.points_count} vectors")
        else:
            print(f"   ❌ Chunks ({CHUNK_COLLECTION}): Missing")

        # Check Entities
        if client.collection_exists(ENTITY_COLLECTION):
            info = client.get_collection(ENTITY_COLLECTION)
            print(f"   ✅ Entities ({ENTITY_COLLECTION}): {info.points_count} vectors")
        else:
            print(f"   ❌ Entities ({ENTITY_COLLECTION}): Missing")
    except Exception as e:
        print(f"   ❌ Connection Failed: {e}")

def check_neo4j():
    print("\n🔍 [Neo4j Health Check]")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            # Count Nodes
            res = session.run("MATCH (n) RETURN count(n) as c")
            count = res.single()["c"]
            print(f"   ✅ Total Nodes: {count}")
            
            # Count Relations
            res = session.run("MATCH ()-[r]->() RETURN count(r) as c")
            rel_count = res.single()["c"]
            print(f"   ✅ Total Relations: {rel_count}")
            
            # Check Orphan Nodes (Quality Check)
            res = session.run("MATCH (n) WHERE NOT (n)--() RETURN count(n) as c")
            orphan_count = res.single()["c"]
            orphan_rate = (orphan_count / count * 100) if count > 0 else 0
            print(f"   ⚠️ Orphan Nodes: {orphan_count} ({orphan_rate:.1f}%)")
            
            if orphan_rate > 50:
                print("      -> Warning: High orphan rate suggests poor connectivity.")
                
        driver.close()
    except Exception as e:
        print(f"   ❌ Connection Failed: {e}")

if __name__ == "__main__":
    print("🏥 Uni-Copilot System Health Check")
    check_qdrant()
    check_neo4j()
    print("\nDone.")