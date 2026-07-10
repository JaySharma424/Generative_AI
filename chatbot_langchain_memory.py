from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory  # Photographic Memory for LLM
from langchain_core.runnables.history import RunnableWithMessageHistory 


load_dotenv()

llm = ChatGoogleGenerativeAI(
  model = "gemini-2.5-flash", 
  temperature = .9
)


# 1. Setup prompt structure with a historical message placeholder
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful chatbot assistant."),
    MessagesPlaceholder(variable_name="history"), #Util also use the memory from the messages history
    ("human", "{input}")
])

# 2. Build conversational line using LCEL
conversational_chain = prompt_template | llm

# 3. Maintain session store externally (In-memory for runtime simulation)
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 4. Wrap the sequence with Chat Message History management
chain_with_history = RunnableWithMessageHistory(
    conversational_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 5. Invoke conversational exchange utilizing session configs
response_1 = chain_with_history.invoke(
    {"input": "Hi, I am Shivan"},
    config={"configurable": {"session_id": "session_id_1"}}
)
print("AI Response 1:", response_1.content)

response_2 = chain_with_history.invoke(
    {"input": "Can you tell my name"},
    config={"configurable": {"session_id": "session_id_1"}}
)
print("AI Response 2:", response_2.content)

