from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Give me a joke on the topic, {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain me the following joke, \n{joke}',
    input_variables=['joke']
)

joke_chain = RunnableSequence(prompt1, model, parser)
exp_chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(joke_chain.invoke({'topic' : 'Harry Potter'}))
print() 
print(exp_chain.invoke({'topic' : 'Harry Potter'}))