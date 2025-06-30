from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',
    encode_kwargs={'normalize_embeddings' : True}
)

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    collection_name='sample',
    embedding_function=embedding,
    persist_directory='./rag_components/vector_stores/chromadb'
)

vector_store.add_documents(docs)

vector_store.get(include=['metadatas', 'documents', 'embeddings'])

res1 = vector_store.similarity_search(
    query='Who is the captain of mumbai indians',
    k=2
)

res2 = vector_store.similarity_search_with_score(
    query='Who is the captain of mumbai indians',
    k=2
)

res3 = vector_store.similarity_search_with_score(
    query="Who is an all-rounder",
    filter={"team": "Chennai Super Kings"}
)

res = [res1, res2, res3]
for i in res:
    for x in i:
        print(x)