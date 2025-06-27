from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.2,
        max_new_tokens=256
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke('Who is Rohit Sharma and what are his achievements in his career?')
print(result)