import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

gemini_key = os.getenv("GOOGLE_API_KEY")

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# print(vars(llm_model))
# print("*" * 200)
# print(llm_model.__dict__())

# print(type(chat_model))

# print(type(True))