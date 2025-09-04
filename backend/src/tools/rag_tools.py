import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from src.config.settings import settings

class RAGTools:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.vector_db_path)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key
        )
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get the collection for BDD code"""
        try:
            self.collection = self.client.get_collection(
                name="bdd_codebase",
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name="bdd_codebase",
                embedding_function=self.embedding_function
            )
    
    def index_codebase(self, features: List[Dict], step_defs: List[Dict]):
        """Index the BDD codebase for RAG"""
        documents = []
        metadatas = []
        ids = []
        
        # Index feature files
        for idx, feature in enumerate(features):
            documents.append(feature['content'])
            metadatas.append({
                'type': 'feature',
                'path': feature['path'],
                'name': feature['name']
            })
            ids.append(f"feature_{idx}")
        
        # Index step definitions
        for idx, step_def in enumerate(step_defs):
            documents.append(step_def['content'])
            metadatas.append({
                'type': 'step_definition',
                'path': step_def['path'],
                'name': step_def['name']
            })
            ids.append(f"step_def_{idx}")
        
        # Add to collection
        if documents:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
    
    def search_similar_code(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar code in the indexed codebase"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        formatted_results = []
        for idx in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][idx],
                'metadata': results['metadatas'][0][idx]
            })
        
        return formatted_results