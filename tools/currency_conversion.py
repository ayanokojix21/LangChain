from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# Tool Creation

@tool
def get_conversion_factor(base_currency: str, target_currency:str) -> float:
    '''
    This Function fetches the currency conversion factor between a given base currency and a target currency
    '''
    
    url = f'https://v6.exchangerate-api.com/v6/289ca12feaddffd62b159c38/pair/{base_currency}/{target_currency}'
    
    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    '''
    Given a currency conversion rate this function calculates the target currency value from a given base currency value
    '''
    
    return base_currency_value * conversion_rate

llm = ChatGroq(model='llama-3.3-70b-versatile')

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between inr and pkr and based on that convert 10 inr to pkr')]

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    
    if tool_call['name'] == 'get_conversion_factor':
        
        tool_message1 = get_conversion_factor.invoke(tool_call['args'])
        conversion_rate = tool_message1['conversion_rate']
        messages.append(ToolMessage(tool_call_id=tool_call["id"], content=json.dumps(tool_message1)))
    
    
    if tool_call['name'] == 'convert':
        
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call['args'])
        messages.append(ToolMessage(tool_call_id=tool_call["id"], content=json.dumps(tool_message2)))
        
result = llm_with_tools.invoke(messages)
print(result.content)
        
