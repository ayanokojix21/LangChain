from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

template1 = PromptTemplate(
    template='Generate me a well written tweet on the topic, {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Generate me a well written Linked-In post on topic, {topic}',
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(template1, model, parser),
    'post' : RunnableSequence(template2, model, parser)
})

result = parallel_chain.invoke({'topic' : 'Rohit Sharma'})
print(f'Tweet --> {result['tweet']}')
print(f'LinkedIn --> {result['post']}')