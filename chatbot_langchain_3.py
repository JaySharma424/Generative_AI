from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()


# LLM Model 

llm = ChatGoogleGenerativeAI(
  model = "gemini-3.5-flash", 
  temperature = 0.7
)

# Review provided in French
review_text = """Les ordinateurs portables GamersTech impressionne par ses
performances exceptionnelles et son design élégant. De sa configuration
matérielle robuste à un clavier RVB personnalisable et un système de
refroidissement efficace, il établit un équilibre parfait entre prouesses
de jeu et portabilité."""

# Step 1: Translate Review to English
translate_prompt = ChatPromptTemplate.from_template("Translate the following review to English:\n\n{Review}")
translation_chain = translate_prompt | llm | StrOutputParser()

# Step 2: Summarize the English Review
summary_prompt = ChatPromptTemplate.from_template("Can you summarize the following review in 1 sentence:\n\n{English_Review}")
summary_chain = summary_prompt | llm | StrOutputParser()

# Step 3: Detect Language of Original Review
language_prompt = ChatPromptTemplate.from_template("What language is the following review:\n\n{Review}")
language_chain = language_prompt | llm | StrOutputParser()

# Step 4: Generate a Follow-up Message in the original language
followup_prompt = ChatPromptTemplate.from_template(
    "Write a follow-up response to the following summary in the specified language:\n\n"
    "Summary: {summary}\n\n"
    "Language: {language}"
)
followup_chain = followup_prompt | llm | StrOutputParser()

# 🔗 Combine steps sequentially using dictionary assignments
# As data moves through the pipe, keys are iteratively added to the operational state map.
overall_chain = (
    RunnablePassthrough.assign(English_Review=translation_chain)
    | RunnablePassthrough.assign(summary=summary_chain)
    | RunnablePassthrough.assign(language=language_chain)
    | RunnablePassthrough.assign(followup_message=followup_chain)
)

# Execute the comprehensive pipeline
output = overall_chain.invoke({"Review": review_text})

print("\n✅ Final Output Map Structure:")
print(output)