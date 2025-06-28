from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

class Feedback(BaseModel):
    
    sentiment : Literal['positive', 'negative'] = Field(description='Classify the sentiment of the feedback')
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of following Feedback into Positive or Negative \n {feedback} \n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction' : parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to a positive feedback \n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to a negative feedback \n{feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda : 'Could not find sentiment')
) 

chain = classifier_chain | branch_chain

feedback = '''
Harry Potter is a magical, unforgettable journey that blends imagination, friendship, and courage. 
J.K. Rowling creates a rich, immersive world full of wonder, mystery, and life lessons. 
With powerful themes and lovable characters, it's a timeless story that continues to inspire readers of all ages.
'''

result = chain.invoke({'feedback' : feedback})
print(result)

chain.get_graph().print_ascii()

