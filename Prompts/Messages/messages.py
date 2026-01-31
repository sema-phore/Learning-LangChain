from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about yourself in 5 lines")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)