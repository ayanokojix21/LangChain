from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate([
    ('system', 'You are an Expert in {domain}. Now help me with questions related to this field.'),
    ('human', 'Explain in Simple terms about {topic}')
])

prompt = template.invoke({'domain' : 'cricket', 'topic' : 'DRS'})

print(prompt)
