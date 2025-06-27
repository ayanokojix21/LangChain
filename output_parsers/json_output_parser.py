from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.9)

parser = JsonOutputParser()

template = PromptTemplate.from_template(
    'Give me Character Name, Group Name and crush of characters from Harry Potter Series. \n{format_instructions}',
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})
print(result)
print(type(result))