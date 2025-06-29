from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

prompt = PromptTemplate(
    template='You are an expert and pro in English, correct the following text, \n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader('rag/document_loader/text_docs.txt', encoding='utf-8')

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'text' : docs[0].page_content})
print(result) 