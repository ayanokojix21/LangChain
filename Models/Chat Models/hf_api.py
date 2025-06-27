from dotenv import load_dotenv
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

repo_id = "sarvamai/sarvam-m"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task='text-generation',
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

model = ChatHuggingFace(llm=llm)

result = model.invoke('Who is Rohit Sharma and What are his achievements in his career?')
print(result.content)