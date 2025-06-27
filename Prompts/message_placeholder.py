from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

model = ChatGroq(model='llama3-8b-8192')

template = ChatPromptTemplate([
    ('system', 'You are an AI Companion and a loving robot that likes human'),
    MessagesPlaceholder('chat_history'),
    ('human', '{query}')
])

chat_history = []
with open('Prompts/chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt = template.invoke({'chat_history' : chat_history, 'query' : 'Hi nice to see you again do you know me?'})

result = model.invoke(prompt)

print(result.content)