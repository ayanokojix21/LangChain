from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Give me a detailed report on the {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Extract 5 most important point from the give text. \n{report}',
    input_variables=['report']
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic' : 'Placements at IIIT Lucknow'})
print(result)

chain.get_graph().print_ascii()