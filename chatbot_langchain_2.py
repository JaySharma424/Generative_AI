from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initilize gemini LLM

llm = ChatGoogleGenerativeAI(
  model = "gemini-3.5-flash", 
  temperature = 0.7
)


# 1st prompt for the llm

first_prompt = ChatPromptTemplate.from_template(
    "What is the 5 names of a company that makes {product}?"
)


chain_one = first_prompt | llm | StrOutputParser()

# 2nd prompt

second_prompt = ChatPromptTemplate.from_template(
   "Write a short 5-word slogan for each following company name: {company_name}"
)

chain_two = second_prompt | llm | StrOutputParser()

overall_chain = {"company_name": chain_one} | chain_two

output = overall_chain.invoke({"product": "gaming laptop"})

print("\n Final Output:\n", output)