from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()  
api_key = os.environ["GROQ_API_KEY"]

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key)