import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 1. Load your .env file where GOOGLE_API_KEY is stored
load_dotenv()



# 2. Initialize the model (it automatically looks for the GOOGLE_API_KEY env variable)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.6
)

# 3. Create your prompt template (matching your lower-case placeholder variable)
prompt = PromptTemplate(
    input_variables=["product"], 
    template="What is a good name for a company that makes {product}?"
)

# 4. Chain the components together
chain = prompt | llm | StrOutputParser()

# 5. Execute the chain and print the result
response = chain.invoke({"product": "mobile"})
print(response)




