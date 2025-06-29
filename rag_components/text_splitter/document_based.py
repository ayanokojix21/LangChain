from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

text = '''
class Calculator
    def __init__(self, a, b)
        self.a = a
        self.b = b

    def add(self):
        return self.a + self.b
    
    def divide(self):
        if self.b = 0:
            return "Can't divide by zero"
        else
        return self.a / self.b

def print_even_numbers(numberList)
    for num in numberList:
        if num % 2 == 0
            print(num)
        else
        print(num + " is odd")

values = [10, 15, 20, "25"]

calc = Calculator(20, 5)
result = calc.divide()
print("Division Result:", result)

print_even_numbers(values)
'''

prompt = PromptTemplate(
    template='Correct the following Python Code, \n{code}',
    input_variables=['code']
)

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

chain = prompt | model | parser

result = []
for chunk in chunks:
    output = chain.invoke({'code' : chunk})
    result.append(output)

for i, res in enumerate(result):
    print(f'\n--- Code {i+1} ---\n{res}\n')
