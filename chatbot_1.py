# import getpass
# api_key = getpass.getpass()

# ✅ Import the Gemini SDK
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# ✅ Configure the SDK with your Gemini API key
genai.configure(api_key=api_key)  # 🔒 Use getpass.getpass() for hidden input in production



model = genai.GenerativeModel("gemini-2.5-flash") # API + What model i want to use

def single_query_chat(user_input):

  prompt = user_input
  response = model.generate_content(prompt) # named it this way
  
  return response.text

# for m in genai.list_models() :  # genai.list_models() gives me list of those model which are present in generativeai module
#   print(m)


  


def chat_with_gemini(user_input):
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Create a custom role-based prompt or give model a persona 
    
    prompt = f"""You are a Senior Data Scientist at Google, specializing in machine learning and data analysis.
    Your expertise includes developing advanced algorithms, creating predictive models, and deriving actionable insights
    from complex datasets. You also communicate technical concepts to non-technical stakeholders and collaborate
    with cross-functional teams. User: {user_input}
    Bot:"""

    # Generate a response using the Gemini model
    response = model.generate_content([prompt])


    return response.text


# print(single_query_chat("tell me about u in 1 line")) 

# print(chat_with_gemini("tell me aout urself in 1 line"))

def run_chatbot() :
  
  print("Welcome to the Gemini Chatbot! Type 'exit' to end the chat.")
  while True:
    user_input = input("Enter your query: ")
    if user_input == 'exit':
      print("Exiting")
      break
    response = chat_with_gemini(user_input)
    print(response)


import gradio as gr

# Create a Gradio interface
def chatbot_interface(user_input):
    return chat_with_gemini(user_input)

# Set up the Gradio interface
iface = gr.Interface(fn=chatbot_interface, inputs="text", outputs="text", title="Gemini Chatbot",
                     description="Chatbot powered by Gemini 2.5 flash. Ask me anything!")
iface.launch(share= True)