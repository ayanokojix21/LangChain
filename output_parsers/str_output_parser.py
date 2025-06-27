from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

template1 = PromptTemplate(
    template='Write a Detailed Report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 Line Summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke('Future of AI Engineers? Will they replace SDE')

print(result)

