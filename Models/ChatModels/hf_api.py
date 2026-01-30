from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# LLM config
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is semaphore stands for, in operating system in 5 points.")

print(result.content)


"""
meta-llama/Llama-3.1-8B-Instruct
openai/gpt-oss-120b
mistralai/Mistral-7B-Instruct-v0.2

"""