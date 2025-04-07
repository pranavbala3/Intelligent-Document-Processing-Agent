import os
import argparse
import sys
from rag import RAGAgent
from settings_service import SettingsService
import google.generativeai as genai
import logging
import time

def configure_logging(verbose=False):
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    
    # Set up root logger for the demo
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if verbose else logging.WARNING)
    root_logger.handlers = []  # Remove any existing handlers
    root_logger.addHandler(console_handler)
    
    # Configure the demo's logger
    demo_logger = logging.getLogger("rag_demo")
    demo_logger.setLevel(logging.INFO)
    
    # Quiet the other loggers
    for logger_name in ["document_parser", "page_processor", "chromadb", "urllib3", "PIL"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
        
    return demo_logger

class RAGDemo:
    def __init__(self, model_name="gemini-2.0-flash", verbose=False):
        self.logger = configure_logging(verbose)

        try:
            genai.configure(api_key=SettingsService().settings.google_api_key)
            self.logger.info("Google Generative AI API configured successfully")
        except Exception as e:
            self.logger.error(f"Failed to configure Google Generative AI API: {e}")
            sys.exit(1)
            
        self.rag_agent = RAGAgent(
            model_name=model_name,
            chroma_db_path="./chroma_db",
            parallel_ingestion=True,
            batch_size=100
        )
        
        self.indexed_documents = []
        
    def print_header(self):
        print("\n" + "=" * 80)
        print(" " * 30 + "INTELLIGENT DOCUMENT AGENT")
        print("=" * 80)
        print("\nThis agent allows you to index documents and query them using RAG.")
        print("Follow the prompts to get started.\n")
        
    def get_documents(self):
        documents = []
        print("\n--- DOCUMENT INDEXING ---")
        print("Enter the paths to the documents you want to index.")
        print("Press Enter with no input when you're done.")
        
        while True:
            doc_path = input("\nDocument path (or Enter to finish): ").strip()
            if not doc_path:
                break

            # Validate the document path
            if not os.path.exists(doc_path):
                print(f"Error: File '{doc_path}' does not exist. Please try again.")
                continue
                
            # Check that the file is a PDF
            _, ext = os.path.splitext(doc_path)
            if ext.lower() != '.pdf':
                print(f"Warning: File '{doc_path}' is not a PDF. This system works best with PDFs.")
                confirm = input("Continue with this file anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
                    
            documents.append(doc_path)
            print(f"Added document: {doc_path}")
            
        return documents
            
    def index_documents(self, documents):
        if not documents:
            print("\nNo documents to index. Moving to query phase.")
            return
            
        print(f"\nIndexing {len(documents)} document(s). This may take some time...")
        start_time = time.time()
        
        try:
            self.rag_agent.index_documents(documents)
            self.indexed_documents.extend(documents)
            
            elapsed_time = time.time() - start_time
            print(f"\nIndexing completed in {elapsed_time:.2f} seconds.")
            print(f"Successfully indexed {len(documents)} document(s).")
        except Exception as e:
            self.logger.error(f"Error during indexing: {e}")
            print(f"\nAn error occurred during indexing: {e}")
            
    def handle_queries(self):
        if not self.indexed_documents:
            print("\nNo documents have been indexed. Please index documents first.")
            return
            
        print("\n--- QUERY MODE ---")
        print("Enter your questions about the documents.")
        print("Press Ctrl+C or Enter with no input to exit.")
        
        try:
            while True:
                query = input("\nYour question (or Enter to exit): ").strip()
                if not query:
                    break
                    
                print("\nProcessing your query...")
                start_time = time.time()
                
                try:
                    result = self.rag_agent.answer_query(query, self.indexed_documents)
                    elapsed_time = time.time() - start_time
                    
                    print("\n" + "-" * 80)
                    print("ANSWER:")
                    print("-" * 80)
                    print(result["response"])
                    print("-" * 80)
                    
                    # Print retrieval details if available
                    print(f"\nQuery processed in {elapsed_time:.2f} seconds.")
                    print(f"Used {result['num_images_used']} page images and {len(result['retrieved_docs'])} text chunks.")
                    
                    # Ask if user wants to see detailed retrieval info
                    show_details = input("\nShow retrieval details? (y/n): ").strip().lower()
                    if show_details == 'y':
                        print("\nRETRIEVAL DETAILS:")
                        print("-" * 80)
                        for i, detail in enumerate(result["retrieval_details"], 1):
                            print(f"{i}. Document: {detail['document']}, Page: {detail['page'] + 1}, Type: {detail['element_type']}")
                            print(f"   Summary: {detail['summary']}")
                            print()
                    
                except Exception as e:
                    self.logger.error(f"Error processing query: {e}")
                    print(f"\nAn error occurred while processing your query: {e}")
                
        except KeyboardInterrupt:
            print("\nExiting query mode.")
        
    def run(self):
        self.print_header()
        
        documents = self.get_documents()
        self.index_documents(documents)
        self.handle_queries()
        
        print("\nExited!")

def main():
    parser = argparse.ArgumentParser(description="RAG Agent")
    parser.add_argument("--model", default="gemini-2.0-flash", 
                        help="Gemini model to use (default: gemini-2.0-flash)")
    args = parser.parse_args()
    
    try:
        demo = RAGDemo(model_name=args.model)
        demo.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        demo.logger.error(f"Unexpected error: {e}")
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()