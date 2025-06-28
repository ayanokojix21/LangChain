from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

template1 = PromptTemplate(
    template='Generate a {length} report on topic, {topic}',
    input_variables=['length', 'topic']
)

template2 = PromptTemplate(
    template='Summarize the following report, \n{report}',
    input_variables=['report']
)

report_chain = RunnableSequence(template1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) >= 500, RunnableSequence(template2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, branch_chain)
result = final_chain.invoke({'length' : 'above 500 words', 'topic' : 'India'})
print(result)
print(len(result.split()))