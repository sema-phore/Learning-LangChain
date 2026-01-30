from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.0-pro')

result = model.invoke("Wht is semaphore means, in operating system")

print(result.content)