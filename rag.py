import google.generativeai as genai
from settings_service import SettingsService
import logging
from document_parser import DocumentParser
import chromadb
from chromadb.utils import embedding_functions
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self, 
                 chroma_db_path="./chroma_db", 
                 model_name="gemini-2.0-flash", 
                 parallel_ingestion: bool = True,
                 cache_dir="./rag_cache",
                 batch_size: int = 100):
        
        # Initialize the generative model name and ingestion mode
        # I made parallel and sequential ingestion modes due to rate limits for the Gemini API causing problems for larger docs 
        self.model_name = model_name
        self.parallel_ingestion = parallel_ingestion
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Track indexed documents to avoid reprocessing
        self.indexed_documents = set()
        self._load_indexed_documents()
        
        # Persistent ChromaDB client and collection
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name="rag-docs",
            embedding_function=default_ef
        )
        
        # Store processed documents: key = document_path, value = doc_state
        self.documents = {}
        # Map friendly names to document_paths
        self.document_names = {}

        genai.configure(api_key=SettingsService().settings.google_api_key)
        self.answer_model = genai.GenerativeModel(
            self.model_name,
            generation_config={"response_mime_type": "text/plain"}
        )
        
        self.parser = DocumentParser()

    def _load_indexed_documents(self):
        index_cache_path = os.path.join(self.cache_dir, "indexed_documents.pickle")
        if os.path.exists(index_cache_path):
            try:
                with open(index_cache_path, "rb") as f:
                    self.indexed_documents = pickle.load(f)
                logger.info(f"Loaded {len(self.indexed_documents)} indexed documents from cache")
            except Exception as e:
                logger.warning(f"Error loading indexed documents cache: {str(e)}")
                self.indexed_documents = set()
        else:
            self.indexed_documents = set()
    
    def _save_indexed_documents(self):
        index_cache_path = os.path.join(self.cache_dir, "indexed_documents.pickle")
        try:
            with open(index_cache_path, "wb") as f:
                pickle.dump(self.indexed_documents, f)
        except Exception as e:
            logger.warning(f"Error saving indexed documents cache: {str(e)}")

    def _get_document_cache_path(self, document_path: str) -> str:
        doc_hash = self._get_document_hash(document_path)
        return os.path.join(self.cache_dir, f"{doc_hash}_state.pickle")
    
    def _get_document_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _load_document_state(self, document_path: str) -> Optional[Any]:
        cache_path = self._get_document_cache_path(document_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading document state from cache: {str(e)}")
        return None
    
    def _save_document_state(self, document_path: str, state: Any):
        cache_path = self._get_document_cache_path(document_path)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.warning(f"Error saving document state to cache: {str(e)}")

    def is_document_indexed(self, document_path: str) -> bool:
        # Check if in cached list
        if document_path in self.indexed_documents:
            return True
        
        # Check in the database
        try:
            results = self.collection.get(
                where={"document_path": document_path},
                limit=1
            )
            is_indexed = len(results.get("ids", [])) > 0
            
            # Update cache if it was actually indexed
            if is_indexed:
                self.indexed_documents.add(document_path)
                self._save_indexed_documents()
                
            return is_indexed
        except Exception as e:
            logger.warning(f"Error checking if document is indexed: {str(e)}")
            return False

    def index_document(self, document_path: str) -> None:
        # Check if already indexed
        if self.is_document_indexed(document_path):
            logger.info(f"Document '{document_path}' is already indexed; skipping.")
            return
        
        logger.info(f"Processing document: {document_path}")
        start_time = time.time()
        
        # Try to load document state from cache
        doc_state = self._load_document_state(document_path)
        if doc_state is None:
            # Process the document if not in cache
            doc_state = self.parser.process_document(document_path)
            # Save to cache for future use
            self._save_document_state(document_path, doc_state)
        
        self.documents[document_path] = doc_state
        
        # Process items in batches to avoid memory issues
        total_indexed = 0
        ids_batch = []
        texts_batch = []
        metadatas_batch = []
        
        for page_idx, page in enumerate(doc_state.extracted_layouts):
            layout_items = page.get("layout_items", [])
            
            # Skip empty pages
            if not layout_items:
                continue
                
            for item_idx, item in enumerate(layout_items):
                layout_id = f"{document_path}_{page_idx}_{item_idx}"
                
                # Skip items with empty summaries
                summary = item.get("summary", "").strip()
                if not summary:
                    continue
                    
                meta = {
                    "document_path": document_path,
                    "page_number": page_idx,
                    "element_type": item.get("element_type", ""),
                    "section": item.get("section", "")
                }
                
                ids_batch.append(layout_id)
                texts_batch.append(summary)
                metadatas_batch.append(meta)
                
                # When batch size is reached, add to collection
                if len(ids_batch) >= self.batch_size:
                    self._add_batch_to_collection(ids_batch, texts_batch, metadatas_batch)
                    total_indexed += len(ids_batch)
                    ids_batch, texts_batch, metadatas_batch = [], [], []
        
        if ids_batch:
            self._add_batch_to_collection(ids_batch, texts_batch, metadatas_batch)
            total_indexed += len(ids_batch)
        
        # Update indexed documents cache
        if total_indexed > 0:
            self.indexed_documents.add(document_path)
            self._save_indexed_documents()
        
        processing_time = time.time() - start_time
        logger.info(f"Indexed {total_indexed} layout items for document '{document_path}' in {processing_time:.2f} seconds")

    def _add_batch_to_collection(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"Error adding batch to collection: {str(e)}")
            for i in range(len(ids)):
                try:
                    self.collection.add(
                        ids=[ids[i]],
                        documents=[texts[i]],
                        metadatas=[metadatas[i]]
                    )
                except Exception as e2:
                    logger.error(f"Error adding item {ids[i]}: {str(e2)}")

    def index_documents(self, document_paths: List[str]) -> None:
        # Load all documents into memory
        for doc_path in document_paths:
            if doc_path not in self.documents:
                doc_state = self._load_document_state(doc_path)
                if doc_state is not None:
                    self.documents[doc_path] = doc_state
                    logger.info(f"Loaded cached document state for: {doc_path}")
    
        # Filter out already indexed documents
        docs_to_index = [doc for doc in document_paths if not self.is_document_indexed(doc)]
        
        if not docs_to_index:
            logger.info("All documents are already indexed.")
            return
            
        logger.info(f"Indexing {len(docs_to_index)} documents out of {len(document_paths)} total")
        
        start_time = time.time()
        
        if self.parallel_ingestion and len(docs_to_index) > 1:
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.index_document, doc_path): doc_path for doc_path in docs_to_index}
                for future in as_completed(futures):
                    doc_path = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error indexing document {doc_path}: {e}")
        else:
            for doc_path in docs_to_index:
                try:
                    self.index_document(doc_path)
                except Exception as e:
                    logger.error(f"Error indexing document {doc_path}: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Finished indexing {len(docs_to_index)} documents in {total_time:.2f} seconds")

    def answer_query(self, query: str, document_paths: List[str]) -> Dict[str, Any]:
        resolved_paths = []
        for doc in document_paths:
            if doc in self.document_names:
                resolved_paths.append(self.document_names[doc])
            else:
                resolved_paths.append(doc)
        
        # Filter condition that retrieves layout items from any of the specified documents
        filter_condition = {"document_path": {"$in": resolved_paths}}
        
        try:
            results = self.collection.query(
                query_texts=[query], 
                n_results=5, 
                where=filter_condition,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_docs = results.get("documents", [[]])[0]
            retrieved_metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            combined_page_numbers = set()
            combined_texts = []
            retrieval_details = []
            
            # Process the retrieved items and their metadata
            for i, (doc, meta, dist) in enumerate(zip(retrieved_docs, retrieved_metas, distances)):
                if "page_number" in meta:
                    combined_page_numbers.add(meta["page_number"])
                    
                # Create retrieval detail record
                retrieval_details.append({
                    "document": os.path.basename(meta.get("document_path", "unknown")),
                    "page": meta.get("page_number", -1),
                    "element_type": meta.get("element_type", "unknown"),
                    "distance": dist,
                    "summary": doc[:100] + "..." if len(doc) > 100 else doc
                })
                
                combined_texts.append(doc)
            
            # For each document, fetch the corresponding images for the retrieved page numbers
            combined_images = []
            for doc_path in resolved_paths:
                if doc_path not in self.documents:
                    logger.warning(f"Document '{doc_path}' not indexed; skipping its images.")
                    continue
                    
                doc_state = self.documents[doc_path]
                for pn in combined_page_numbers:
                    if pn < len(doc_state.pages_as_base64_jpeg_images):
                        combined_images.append(doc_state.pages_as_base64_jpeg_images[pn])
            
            logger.info(f"Answering query: {query}")
            logger.info(f"Retrieved {len(combined_texts)} relevant text chunks")
            logger.info(f"Retrieved {len(combined_images)} relevant images")
            
            # Build messages for the generative model
            messages = []
            
            # Add context text
            context_text = "Based on the following information from the documents:\n\n"
            for i, text in enumerate(combined_texts, 1):
                context_text += f"[{i}] {text}\n\n"
            messages.append({"text": context_text})
            
            # Add images if available
            for img in combined_images:
                messages.append({"mime_type": "image/jpeg", "data": img})
            
            # Append the query
            messages.append({"text": f"Answer the following question using the provided context: {query}"})
            
            response = self.answer_model.generate_content(messages)
            
            return {
                "response": response.text,
                "retrieved_docs": combined_texts,
                "page_numbers": list(combined_page_numbers),
                "retrieval_details": retrieval_details,
                "num_images_used": len(combined_images)
            }
            
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return {
                "response": f"Error processing query: {str(e)}",
                "retrieved_docs": [],
                "page_numbers": [],
                "retrieval_details": [],
                "error": str(e)
            }
    
    def register_document(self, friendly_name: str, document_path: str) -> None:
        self.document_names[friendly_name] = document_path
        logger.info(f"Registered '{friendly_name}' for document: {document_path}")
        
    def get_document_info(self) -> List[Dict[str, Any]]:
        info = []
        for name, path in self.document_names.items():
            is_indexed = path in self.indexed_documents
            info.append({
                "name": name,
                "path": path,
                "indexed": is_indexed
            })
        return info

if __name__ == "__main__":
    agent = RAGAgent(
    chroma_db_path="./chroma_db",
    model_name="gemini-2.0-flash",
    parallel_ingestion=True,
    batch_size=100
    )

    document_paths = [
        "testing/GaLore.pdf",
        "testing/LoRA-Pro.pdf"
    ]

    # Index documents
    agent.index_documents(document_paths)

    # Answer queries
    result = agent.answer_query("Give me the key details from Figure 2 in the GaLore paper?", document_paths)
    print(result["response"])