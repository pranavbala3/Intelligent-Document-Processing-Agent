# Intelligent Document Processing Agent
This is a Intelligent Document Processing Agent designed to efficiently process and analyze research papers in PDF format. 

## Architecture and Approach
The architecture primarily consists of three modular components:

1. **Document Processing Module**: Responsible for ingesting and parsing PDF documents.
2. **Information Extraction Module**: Utilizes NLP techniques to extract structured information such as titles, abstracts, sections, tables, and references.
3. **Query Processing Module**: Implements a Retrieval-Augmented Generation (RAG) model to handle intelligent queries by retrieving relevant context and generating responses based on user inputs.

### Rationale on Why I Used a RAG-Based Approach:
This came down to a few main points:

1. By retrieving relevant document segments before generation, RAG significantly reduces hallucinations and factual errors compared to strictly generative approaches.
2. RAG enables traceability between generated answers and source material, improving user trust and facilitating verification.
3. Rather than processing entire documents for each query, RAG identifies and processes only the most relevant sections, reducing computational overhead.

## Information Extraction and Query Processing Steps

### Information Extraction
- PDFs are processed using `page_processor.py` and `document_parser.py`. The pipeline extracts text and preserves structural components such as tables, figures and references.
- This is a hybrid approach: First, use text extraction libraries to extract text from documents. Then, identify whether additional processing is needed based on certain characteristics:
  *   detection of grid-like structure via computer vision techniques
  *   whitespace pattern analysis
  *   mentioning of figures/tables in the text
  *   text density
- For pages that require additional processing:
  *   PDF pages are converted to high-quality JPEG images
  *   The Gemini model analyzes visual elements including tables, figures, and complex layouts
  *   Structured JSON output identifies all layout elements with descriptions
- Pages are processed parallely using `ProcessPoolExecutor`.

### Query Processing
- The agent uses a RAG approach implemented in `rag.py`. This module indexes extracted information and uses embeddings to retrieve relevant content.
- The agent performs a semantic search using the ChromaDB collection to identify the most relevant document segments for the query. This is achieved by
  * Creating a filter condition to limit the search to the specified documents
  * Executing a vector similarity search to find the top 5 most relevant text chunks
  * Retrieving documents, metadata, and similarity scores
- The agent then processes the retrieved results to prepare them for the generative model. For "complex" context, the agent retrieves relevant document images corresponding to the pages where information was found.
- Finally, the agent constructs a multimodal context for the generative model that includes the text from the retrieved documents, relevant images, and the query.

## Evaluation
To evaluate the performance of the agent, I first tested the document parsing and query processing steps separately. I then manually tested different prompts such as comparisons, summarizations, and evaluation result extractions and checked whether the agent was returning high similarity documents and the correct figures, tables, and/or images.

## Challenges and Solutions
- **Challenge**: Efficiently parsing and extract information from documents. Initially, I had made a structure that involved taking images of the pages of the document and then using a LLM to label sections as figures, tables, abstract, introduction, methods, etc. This was also causing rate limiting problems with the Gemini API. 
  - **Solution**: Implementing a hybrid approach that extracts text and only calls the LLM to help in labelling if the page is complex. Additionally, I made the process of extracting information less granular by simply having the model label information as figures, tables, images, or text. The rate limiting problem is still a problem with larger documents as I have been using the Free Tier.

- **Challenge**: Processing large documents led to memory constraints, particularly when embedding and storing numerous text chunks simultaneously.
  - **Solution**: Implemeting a batched processing approach with configurable batch sizes to control memory usage during document indexing.

- **Challenge**: Initially, the system would reprocess documents each time they were used, even if they had not changed, causing unnecessary computational overhead and API costs.
  - **Solution**: Implementing a comprehensive caching system with components to document state caching to store processed document information, hashing to identify document changes efficiently, an indexed document tracking system to avoid reprocessing.

## Things to Work On
- **Long time to index documents**: Indexing documents into the vector database is quite time-consuming and I did not find a good work around to this. I tried to make the indexing parallel too but still takes about 3-4 mins to process a doucment ~15-20 pages. However, that is why I implemented a persistent ChromaDB client and caching to prevent the need for documents to be repeatedly processed if the user has already inputted a document prior. So although indexing takes time, this is a "one-and-done" process. As a result, user prompts are being answer in 5-6 seconds.

## Instructions for Setup and Running the Agent

### Setup

This code was written using Python 3.10.4

1. **Virtual Environment Setup**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
One of the libraries (pdf2image) requires Poppler.
For MacOS:
```bash
brew install poppler
```

For Windows: follow instructions on the [pdf2image docs](https://pdf2image.readthedocs.io/en/latest/installation.html).


2. **Environment File**

View the `.env.template` and create a `.env` file with your Gemini API key.

3. **Run Agent**
```bash
python demo.py
```

This will start the agent and give you further steps on how to use the agent.

## Demo:
https://github.com/user-attachments/assets/d9730362-5310-4312-9e71-7c11e8d726c3
