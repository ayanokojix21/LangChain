from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGroq(model='llama-3.3-70b-versatile')

model2 = ChatGroq(model='llama-3.1-8b-instant')

prompt1 = PromptTemplate(
    template='Give me a short and simple notes on the following text \n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate me a 10 Question Quiz on this text \n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz to form a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

text = '''
🧠 What is an LLM (Large Language Model)?
A Large Language Model (LLM) is an artificial intelligence model trained on massive amounts of text data to understand, generate, and interact with human language. LLMs are a type of deep learning model, typically built on transformer architectures, and are capable of performing a wide range of Natural Language Processing (NLP) tasks such as:

Text generation

Summarization

Translation

Question answering

Dialogue systems (chatbots)

Code generation

Information extraction

🏗️ How Do LLMs Work?
LLMs work by predicting the next word/token in a sentence, given the previous context. They use the Transformer architecture, introduced by Vaswani et al. in 2017, which enables the model to learn long-range dependencies and contextual relationships efficiently using self-attention mechanisms.

🔄 Key Mechanism: Autoregressive Generation
Most LLMs are autoregressive, meaning they generate text one token at a time, using the previous tokens as context. This allows them to generate coherent paragraphs, code, or answers to prompts.

🧱 Key Components of LLMs
Component	Role
Tokenizer	Converts text into numerical tokens
Embedding Layer	Maps tokens to high-dimensional vectors
Transformer Blocks	Capture contextual relationships using attention mechanisms
Output Layer	Converts processed vectors back to tokens

📊 Types of LLM Architectures
Type	Description	Examples
Decoder-Only	Good for text generation, autoregressive tasks	GPT-3, GPT-4, LLAMA, Mistral
Encoder-Only	Good for understanding tasks (e.g., classification, QA)	BERT, RoBERTa
Encoder-Decoder	Good for seq2seq tasks like translation or summarization	T5, BART

🚀 Popular LLMs
Model	Creator	Notable Features
GPT-3 / GPT-4	OpenAI	Large-scale generation, ChatGPT backbone
Claude	Anthropic	Constitutional AI, aligned generation
Gemini	Google DeepMind	Multimodal support, integrated search
LLaMA 2 / 3	Meta	Open-weight models, efficient training
Mistral / Mixtral	Mistral AI	Sparse Mixture-of-Experts (MoE) models
T5	Google	Text-to-text unified format
BERT	Google	Encoder-only, widely used for understanding

📚 Training of LLMs
LLMs are trained in two major phases:

1. Pretraining
Objective: Predict the next token or fill-in-the-blank (masked language modeling).

Trained on: Web data, books, Wikipedia, code, forums, etc.

Result: Model learns general language understanding and generation capabilities.

2. Fine-Tuning
Objective: Specialize the model for tasks (e.g., dialogue, summarization).

Techniques:

Supervised fine-tuning

Reinforcement learning with human feedback (RLHF)

Instruction tuning (e.g., FLAN, Alpaca)

🧠 Capabilities of LLMs
Capability	Description
Natural Language Understanding	Interprets context, sentiment, intent
Text Generation	Writes coherent, contextual, and creative text
Code Generation	Writes functions, scripts, or entire apps (e.g., Copilot)
Translation	Converts text between languages accurately
Multi-Turn Dialogue	Carries conversations with memory (chatbots)
Question Answering	Answers queries from documents or knowledge base

🔒 Limitations of LLMs
Limitation	Description
Hallucination	Generates incorrect or fabricated facts
Context Limit	Limited memory window (context length constraint)
Bias	Inherits bias from training data
Cost & Compute	Requires high GPU/TPU power and memory
Lack of Reasoning	Can struggle with complex logic or math

🧠 Applications of LLMs
Application	Description
Chatbots / Assistants	AI like ChatGPT, Claude, Gemini etc.
Search + RAG	LLM with vector search and retrieval
Autonomous Agents	LLMs used for planning and executing tasks
Dev Tools	Copilot, Codex, Ghostwriter
Education	Personalized tutoring, explanation
Content Creation	Blogs, marketing, social media automation

🔁 LLM vs Chat Model
Feature	LLM (Text Completion)	Chat Model (Conversational)
Input	Plain string	Role-based message list
Role Understanding	❌ No	✅ Yes (system, user, assistant)
Multi-turn Support	❌ No	✅ Yes
Common Use	Generation, summarization	Chatbots, assistants

🧩 LLM Integration Tools
Tool / Framework	Use Case
LangChain	Build LLM apps with memory, tools, chains
Haystack	QA, RAG pipelines for search
Transformers (HF)	Model loading & generation
LangGraph	Multi-agent workflows for LLMs
FastChat	Serve chat models via API

🧠 Summary
LLMs are deep learning models trained on large text corpora.

They are capable of natural language understanding and generation.

Used in various NLP tasks including chat, summarization, code, and search.

Foundation models behind tools like ChatGPT, Claude, Gemini, and more.

Let me know if you'd like:

A simplified beginner version

A PDF version for print

Or modular breakdowns (e.g., only on transformers, training, or applications).
'''

chain = parallel_chain | merge_chain
print(chain.invoke({'text' : text}))

chain.get_graph().print_ascii()