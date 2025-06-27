from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

# Schema
class Review(TypedDict):
    
    key_themes : Annotated[list[str], 'Write down all the key themes discussed in the review in the theme']    
    summary : Annotated[str, 'A Brief Summary of the review']
    sentiment : Annotated[str, 'Return the sentiment of review in any of the 5 categories like Extremely Positive, Positive, Neutral, Negative, Extremely Negative']
    star : Annotated[int, 'Give rating of the review from 1-5']
    pros : Annotated[Optional[list[str]], 'Write down all the pros inside the list']
    cons : Annotated[Optional[list[str]], 'Write down all the cons inside the list']
    name : Annotated[Optional[str], 'Write the Name of the Reviewer']
    
        
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nishchal Chandel
""")

print(result)