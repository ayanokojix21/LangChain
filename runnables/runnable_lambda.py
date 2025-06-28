from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a Joke about topic, {topic}',
    input_variables=['topic']
)

joke_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'length' : RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

result = final_chain.invoke({'topic' : 'IIIT Lucknow'})
print(f'Joke : {result['joke']}, \nWord Count : {result['length']}')