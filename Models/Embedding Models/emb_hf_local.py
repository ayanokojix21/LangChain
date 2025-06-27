from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = 'Delhi is the capital of India'

docs = [
    'Narendra Modi is President of India',
    'Rohit Sharma is Captain of Indian Cricket Team',
    'India is the best country in the world'
]

vector_query = embedding.embed_query(text)
print(str(vector_query))

vector_docs = embedding.embed_documents(docs)
print(str(vector_docs))