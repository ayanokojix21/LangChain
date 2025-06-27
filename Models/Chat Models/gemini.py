from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv('GOOGLE_API_KEY')

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=key)

result = model.invoke('Who is Rohit Sharma and what are his achievements in his career?')
print(result.content)