from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

prompt = PromptTemplate(
    template='You are excellent in PDF summarization then summarize the following pdf, {pdf}',
    input_variables=['pdf']
)

loader = PyPDFLoader('rag/document_loader/pdf_docs.pdf')

docs = loader.load()

chain = prompt | model | parser
result = chain.invoke({'pdf' : docs})
print(result)