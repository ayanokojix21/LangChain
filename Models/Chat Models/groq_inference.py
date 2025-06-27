from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.1-8b-instant')

result = model.invoke('Who is Rohit Sharma and what are his achievements in his career?')
print(result.content)

