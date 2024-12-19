import os
import openai
import requests
import PyPDF2
import streamlit as st
import json

# File path for saving chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Use the Sambanova API for Qwen 2.5-72B-Instruct and Meta-Llama-3.2-1B-Instruct
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

# Estimate token count for messages
def estimate_token_count(messages):
    token_count = 0
    for msg in messages:
        if "content" in msg:
            token_count += len(msg["content"].split()) * 4  # Approximate token count: 4 tokens per word
    return token_count

# Streamlit UI setup
st.set_page_config(page_title="Chatbot with PDF (Botify)", layout="centered")
st.title("Botify")

# Upload a PDF file
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! I am Botify, your assistant. How can I assist you today?"}]
    st.session_state.selected_model = "Qwen2.5-72B-Instruct"

# Button to start a new chat
if st.button("Start New Chat"):
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! Starting a new conversation. How can I assist you today?"}]
    st.session_state.chat_history.append(st.session_state.current_chat)
    st.success("New chat started!")

# Model selection
model_choice = st.selectbox("Select the LLM model:", ["Sambanova (Qwen 2.5-72B-Instruct)", "Sambanova (Meta-Llama-3.2-1B-Instruct)"])

# Change model logic
if model_choice != st.session_state.selected_model:
    st.session_state.selected_model = model_choice
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! I am Botify, your assistant. How can I assist you today?"}]
    st.session_state.chat_history.append(st.session_state.current_chat)

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

# Wait for user input (only process when the user clicks the "Submit" button)
user_input = st.text_input("Your message:", key="user_input", placeholder="Type your message here")
submit_button = st.button("Submit", key="submit_button")

if submit_button and user_input:
    # Add user input to current chat
    st.session_state.current_chat.append({"role": "user", "content": user_input})

    # If a PDF is uploaded, extract its content
    if pdf_file:
        text_content = extract_text_from_pdf(pdf_file)
        # Use the extracted PDF content as part of the prompt
        prompt_text = f"Document content:\n{text_content}\n\nUser question: {user_input}\nAnswer:"
    else:
        # Only use user input for the prompt if no PDF is uploaded
        prompt_text = f"User question: {user_input}\nAnswer:"

    # Add system message with the prompt (for LLM)
    st.session_state.current_chat.append({"role": "system", "content": prompt_text})

    # Adjust the context length for each model
    model = "Qwen2.5-72B-Instruct" if st.session_state.selected_model == "Sambanova (Qwen 2.5-72B-Instruct)" else "Meta-Llama-3.2-1B-Instruct"
    context_length = 8192 if model == "Qwen2.5-72B-Instruct" else 16384

    # Estimate token count and truncate if necessary
    total_tokens = estimate_token_count(st.session_state.current_chat)
    if total_tokens > context_length:
        # Trim the chat history to fit within the token limit, keeping only the last 5 messages
        st.session_state.current_chat = st.session_state.current_chat[-5:]

    remaining_tokens = context_length - estimate_token_count(st.session_state.current_chat)
    max_tokens = min(max(remaining_tokens, 1), 1024)  # Cap max tokens to prevent overly long responses

    try:
        # Call the API to get a response from the selected model
        response = SambanovaClient(
            api_key=sambanova_api_key,
            base_url="https://api.sambanova.ai/v1"
        ).chat(
            model=model,
            messages=st.session_state.current_chat,
            temperature=0.1,
            top_p=0.1,
            max_tokens=max_tokens
        )
        if 'choices' in response and response['choices']:
            answer = response['choices'][0]['message']['content'].strip()
            # Append the model's response to the conversation history
            st.session_state.current_chat.append({"role": "assistant", "content": answer})
            # Save chat history
            save_chat_history(st.session_state.chat_history)
        else:
            st.error("Error: Empty response from the model.")
    except Exception as e:
        st.error(f"Error while fetching response: {e}")

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
