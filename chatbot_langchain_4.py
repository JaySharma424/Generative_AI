from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel #,Parallel Execution
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()


# LLM 

llm = ChatGoogleGenerativeAI(
  model = "gemini-2.5-flash", 
  temperature = 0.7
)

candidate_profile = """
Python Developer
5 years of experience
Python
AWS
LangChain
Generative AI
Docker
"""

jobsearch_prompt = ChatPromptTemplate.from_template("Search top 5 relevant jobs in naukri.com in accordance to the {candidate_profile}")
jobsearch_chain = jobsearch_prompt | llm | StrOutputParser()

quesprep_prompt = ChatPromptTemplate.from_template("Prepare 5 questions per job in the {list_jobs}")
quesprep_chain = quesprep_prompt | llm | StrOutputParser()

intro_prompt = ChatPromptTemplate.from_template("Prepare a 30 word intro per job in the {list_jobs}")
intro_chain = intro_prompt | llm | StrOutputParser()

parallel_agent = RunnableParallel({"interview_question": quesprep_chain, "intro_30": intro_chain})

overall_chain = (
    RunnablePassthrough.assign(list_jobs=jobsearch_chain)
    | RunnablePassthrough.assign(prep_guide=parallel_agent)
    )

result = overall_chain.invoke({"candidate_profile": candidate_profile})

print("\n✅ Final Output Map Structure:")
print(result)

