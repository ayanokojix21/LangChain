from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
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

exp_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explaination' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_chain, exp_chain)

result = final_chain.invoke({'topic' : 'Harry Potter'})
print(f'Joke --> {result['joke']}')
print()
print(f'Explaination --> {result['explaination']}')
