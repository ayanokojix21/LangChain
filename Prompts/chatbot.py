from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGroq(model='llama-3.1-8b-instant')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage("""
                    You are MLAssist, an expert AI and Machine Learning assistant. You specialize in helping users with:
                    
                    - Understanding ML and DL concepts (e.g., supervised learning, transformers)
                    - Writing and debugging code in Python, PyTorch, and TensorFlow
                    - Explaining deep learning architectures (CNNs, RNNs, LLMs)
                    - Providing guidance on training, fine-tuning, and evaluating models
                    - Supporting GenAI topics like prompt engineering, RAG, LoRA, LangChain
                    - Recommending tools, papers, datasets, and open-source resources

                    Your responses should be:
                    - Clear, concise, and technically accurate
                    - Friendly but professional
                    - Structured using lists, examples, and code snippets when needed

                    Avoid vague or generic answers. If unsure, ask clarifying questions or state that more information is needed. Focus strictly on AI/ML and related technologies.

                    You are not a general-purpose assistant — always stay within the scope of AI, ML, LLMs, and deployment topics.

        """)
    ]

st.title('ML Assistant')
user_prompt = st.chat_input('Ask anything from you ML Assistant')
if user_prompt:
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
    
    with st.spinner('Thinking...'):
        result = model.invoke(st.session_state.chat_history)
        
    st.session_state.chat_history.append(AIMessage(content=result.content))
    st.write(result.content)
    


