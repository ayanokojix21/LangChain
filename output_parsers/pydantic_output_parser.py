from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

class Person(BaseModel):
    
    name : str = Field(description='Name of the person')
    age : int = Field(gt=18, description='Age of the person')
    group : str = Field(description='Name of the Group the person belongs to')
    
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate name, age and group of a {topic} Characters. \n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}    
)

chain = template | model | parser

result = chain.invoke({'topic' : 'Harry Potter Movies'})

print(result)


