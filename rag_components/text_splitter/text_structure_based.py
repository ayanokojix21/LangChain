from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

url = 'https://en.wikipedia.org/wiki/2024_Men%27s_T20_World_Cup_final#:~:text=It%20was%20played%20between%20South%20Africa%20and%20India.&text=India%20won%20the%20toss%20and,second%20T20%20World%20Cup%20title.'

loader = WebBaseLoader(url)
docs = loader.load()
text = docs[0].page_content

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=0,
    keep_separator=''
)

chunks = splitter.split_text(text)

prompt = PromptTemplate(
    template='Generate me an interesting post on the following match, {match}',
    input_variables=['match']
)

chain = prompt | model | parser

result = []
for chunk in chunks:
    output = chain.invoke({'match' : chunk})
    result.append(output)

for i, res in enumerate(result):
    print(f'\n--- Post {i+1} ---\n{res}\n')

