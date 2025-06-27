from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv

model = ChatGroq(model='llama-3.3-70b-versatile')

schema = [
    ResponseSchema(name='fact_1', description='Face 1 about the topic'),
    ResponseSchema(name='fact_2', description='Face 2 about the topic'),
    ResponseSchema(name='fact_3', description='Face 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about the {topic}. \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic' : 'Harry Potter Series'})

print(result)