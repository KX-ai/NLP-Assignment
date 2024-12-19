import os
import openai
import requests
import PyPDF2
import streamlit as st
import json

# File path for saving chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Use the Sambanova API for Qwen 2.5-72B-Instruct
class SambanovaClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        openai.api_key = self.api_key
        openai.api_base = self.base_url

    def chat(self, model, messages, temperature=0.7, top_p=1.0, max_tokens=500):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            return response
        except Exception as e:
            raise Exception(f"Error while calling Sambanova API: {str(e)}")

# Use the Together API for Wizard LM-2 (8x22b)
class TogetherClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.together.xyz/v1/chat/completions"

    def chat(self, model, messages):
        payload = {
            "model": model,
            "messages": messages
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            return response.json()
        except Exception as e:
            raise Exception(f"Error while calling Together API: {str(e)}")

# Function to extract text from PDF using PyPDF2
@st.cache_data
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to load chat history from a JSON file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    else:
        return []

# Function to save chat history to a JSON file
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Streamlit UI setup
st.set_page_config(page_title="Chatbot with PDF (Botify)", layout="centered")
st.title("Botify")

# Upload a PDF file
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! I am Botify, your assistant. How can I assist you today?"}]

# Button to start a new chat
if st.button("Start New Chat"):
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! Starting a new conversation. How can I assist you today?"}]
    st.session_state.chat_history.append(st.session_state.current_chat)
    st.success("New chat started!")

# Display chat dynamically
st.write("### Chat Conversation")
for msg in st.session_state.current_chat:
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        if msg["role"] == "user":
            st.markdown(f"**\U0001F9D1 User:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**\U0001F916 Botify:** {msg['content']}")
    else:
        st.error("Error: A message is missing or malformed in the chat history.")

# API keys
sambanova_api_key = st.secrets["general"]["SAMBANOVA_API_KEY"]
together_api_key = "db476cc81d29116da9b75433badfe89666552a25d2cd8efd6cb5a0c916eb8f50"

# Model selection
model_choice = st.selectbox("Select the LLM model:", [
    "Sambanova (Qwen 2.5-72B-Instruct)",
    "Together (Wizard LM-2 8x22b)",
    "Meta-Llama (Meta-Llama-3.2-1B-Instruct)"
])

# Wait for user input
user_input = st.text_input("Your message:", key="user_input", placeholder="Type your message here and press Enter")
submit_button = st.button("Submit")

if submit_button and user_input:
    st.session_state.current_chat.append({"role": "user", "content": user_input})

    if pdf_file:
        text_content = extract_text_from_pdf(pdf_file)
        prompt_text = f"Document content:\n{text_content}\n\nUser question: {user_input}\nAnswer:"
    else:
        prompt_text = f"User question: {user_input}\nAnswer:"

    st.session_state.current_chat.append({"role": "system", "content": prompt_text})

    try:
        if model_choice == "Sambanova (Qwen 2.5-72B-Instruct)":
            response = SambanovaClient(
                api_key=sambanova_api_key,
                base_url="https://api.sambanova.ai/v1"
            ).chat(
                model="Qwen2.5-72B-Instruct",
                messages=st.session_state.current_chat,
                temperature=0.1,
                top_p=0.1,
                max_tokens=8192  # 8k tokens for Qwen
            )
            answer = response['choices'][0]['message']['content'].strip()

        elif model_choice == "Together (Wizard LM-2 8x22b)":
            response = TogetherClient(api_key=together_api_key).chat(
                model="wizardlm2-8x22b",
                messages=st.session_state.current_chat
            )
            answer = response.get('choices', [{}])[0].get('message', {}).get('content', "No response received.")
        
        elif model_choice == "Meta-Llama (Meta-Llama-3.2-1B-Instruct)":
            # Adjust the API call for Meta-Llama-3.2-1B-Instruct
            response = SambanovaClient(
                api_key=sambanova_api_key,
                base_url="https://api.sambanova.ai/v1"
            ).chat(
                model="Meta-Llama-3.2-1B-Instruct",  # Meta-Llama model name
                messages=st.session_state.current_chat,
                temperature=0.7,
                top_p=1.0,
                max_tokens=16000  # 16k tokens for Meta-Llama
            )
            answer = response['choices'][0]['message']['content'].strip()

        st.session_state.current_chat.append({"role": "assistant", "content": answer})
    
    except Exception as e:
        st.error(f"Error while fetching response: {e}")

# Save chat history
save_chat_history(st.session_state.chat_history)

# Display chat history with deletion option
with st.expander("Chat History"):
    for i, conversation in enumerate(st.session_state.chat_history):
        with st.container():
            st.write(f"**Conversation {i + 1}:**")
            for msg in conversation:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    role = "User" if msg["role"] == "user" else "Botify"
                    st.write(f"**{role}:** {msg['content']}")
                else:
                    st.error(f"Error: Malformed message in conversation {i + 1}.")
            if st.button(f"Delete Conversation {i + 1}", key=f"delete_{i}"):
                del st.session_state.chat_history[i]
                save_chat_history(st.session_state.chat_history)
                st.experimental_rerun()
