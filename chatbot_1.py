import getpass
api_key = getpass.getpass()

# ✅ Import the Gemini SDK
import google.generativeai as genai

# ✅ Configure the SDK with your Gemini API key
genai.configure(api_key=api_key)  # 🔒 Use getpass.getpass() for hidden input in production



model = genai.GenerativeModel("gemini-3.5-flash") # API + What model i want to use

def single_query_chat(user_input):

  prompt = user_input
  response = model.generate_content(prompt) # named it this way

# for m in genai.list_models() :  // genai.list_models() gives me list of those model which are present in generativeai module
#   print(m)


  return response.text


def chat_with_gemini(user_input):
    model = genai.GenerativeModel('gemini-3.5-flash')

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
run_chatbot()