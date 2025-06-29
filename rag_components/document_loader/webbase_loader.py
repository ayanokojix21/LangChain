from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

urls = [
    'https://en.wikipedia.org/wiki/2024_Men%27s_T20_World_Cup_final#:~:text=It%20was%20played%20between%20South%20Africa%20and%20India.&text=India%20won%20the%20toss%20and,second%20T20%20World%20Cup%20title.'
]

loader = WebBaseLoader(urls)

docs = loader.load()

prompt = PromptTemplate(
    template='Give me summary of the following match in a crazy and exicing way tell me who won and important players in match and their contributions never hallucinate, \n{match}',
    input_variables=['match']
)

chain = prompt | model | parser
result = chain.invoke({'match' : docs[0].page_content})

print(result)