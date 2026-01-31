from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

# Declare the Model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature= 0.5,
    do_sample=True
)
model = ChatHuggingFace(llm=llm)

# StreamLit
# Heading
st.header('Research Tool')

"""
# Taking Input from user
# This is static prompt
#user_input_static = st.text_input('Enter your prompt')
"""

# Dynamic Input
# Creating Dropdowns
paper_input = st.selectbox(
    "Select the research paper name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select the Explanation Style",
    [
        "Beginer-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathemetical"
    ]
)

length_input = st.selectbox(
    "Select the Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

# Prompt template
template = load_prompt("Prompts/templete.json")

"""
# Fill the placeholders
prompt = template.invoke(
    {
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    }
)
"""

# Create a button
if st.button('Summarize'):
    # Creating a chain with template and model
    chain = template | model
    result = chain.invoke(
        {
            'paper_input': paper_input,
            'style_input': style_input,
            'length_input': length_input
        }
    )
    st.write(result.content)