import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from typing import List, Dict
from src.agents.bdd_generator_agent import BDDGeneratorAgent
from src.config.settings import settings
from src.config.logging import get_logger

logger = get_logger(__name__)

class RAGTools:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.vector_db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embedding_function = None
        # Prefer HF Inference API if key is present, otherwise fall back to local embeddings
        hf_key = getattr(settings, 'huggingface_api_key', None)
        if hf_key:
            try:
                # Explicitly specify a widely available sentence-transformers model for the Inference API
                hf_func = embedding_functions.HuggingFaceEmbeddingFunction(
                    api_key=hf_key,
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                # Preflight a tiny request to catch JSONDecodeError/non-JSON responses early
                _ = hf_func(input=["healthcheck"])
                self.embedding_function = hf_func
                logger.debug("Using Hugging Face Inference API for embeddings.")
            except Exception as e:
                logger.warning(
                    "Hugging Face embedding preflight failed; falling back to local SentenceTransformer. Error: %s",
                    str(e)
                )
        if self.embedding_function is None:
            # Local embeddings avoid network/API errors; model downloads on first use
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.debug("Using local SentenceTransformer embeddings.")
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get the collection for BDD code"""
        try:
            self.collection = self.client.get_collection(
                name="bdd_codebase",
                embedding_function=self.embedding_function
            )
            logger.debug(f'Collection \'bdd_codebase\' in ChromaDB already exists. Using the existing collection')
        except:
            logger.debug(f'Collection \'bdd_codebase\' in ChromaDB doesn\'t exists. Creating a new collection')
            self.collection = self.client.create_collection(
                name="bdd_codebase",
                embedding_function=self.embedding_function
            )
    
    def index_codebase(self, features: List[Dict], step_defs: List[Dict], llmAgent: BDDGeneratorAgent):
        """Index the BDD codebase for RAG"""

        documents = []
        metadatas = []
        ids = []
        
        # Index feature files
        for idx, feature in enumerate(features):
            description = llmAgent.generate_description_of_file(feature['path'], feature['content'])
            documents.append(description)
            metadatas.append({
                'type': 'feature',
                'path': feature['path'],
                'content': feature['content']
            })
            ids.append(feature['name'])
        
        # Index step definitions
        for idx, step_def in enumerate(step_defs):
            description = llmAgent.generate_description_of_file(step_def['path'], step_def['content'])
            documents.append(description)
            metadatas.append({
                'type': 'step_definition',
                'path': step_def['path'],
                'content': step_def['content']
            })
            ids.append(step_def['name'])
        
        # Add to collection
        if documents:
            try:
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                # Most common cause: HF Inference API returned non-JSON (e.g., 401/403/5xx or HTML)
                logger.error(
                    "Failed to upsert into Chroma collection due to embedding error. "
                    "This often indicates an invalid/missing Hugging Face API key, an unavailable model, "
                    "or a non-JSON response from the HF Inference API. Error: %s", str(e)
                )
                raise
        logger.info(f'Indexing completed for \'bdd_codebase\' for RAG')
    
    def search_similar_code(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar code in the indexed codebase"""
        if(self.collection.count() == 0):
            logger.warning("The collection is empty. No documents to search.")
            return []
        
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