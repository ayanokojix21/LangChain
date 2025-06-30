from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

    
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-small-en-v1.5',
    encode_kwargs={'normalize_embeddings' : True}
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name="chroma_docs",  
    persist_directory="./rag_components/retriever/chromadb"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = 'What is Chroma used for?'
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
    
