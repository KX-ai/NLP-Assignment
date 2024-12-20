import os
import openai
import requests
import PyPDF2
import streamlit as st
import json
from groq import Groq

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

# Updated function to transcribe audio using the Groq API
def transcribe_audio(file):
    whisper_api_key = st.secrets["whisper"]["WHISPER_API_KEY"]  # Access Whisper API key (Groq API key)
    url = "https://api.groq.com/openai/v1/audio/transcriptions"  # Groq transcription endpoint

    # Specify the transcription model (Groq's API model)
    model = "whisper-1"  # Replace this with the actual model Groq uses for transcription

    # Prepare the headers for the API request
    headers = {
        "Authorization": f"Bearer {whisper_api_key}",  # Authorization with your API key
    }

    # Prepare the data payload for the request
    data = {
        "model": model,  # Add the model field to the request
        "language": "en",  # Specify the language
    }

    try:
        # Open the audio file and send it as binary content
        with file:
            files = {"file": file.getvalue()}  # Get the file content as binary
            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data  # Send additional data (model and language)
            )

        # Check if the request was successful
        if response.status_code == 200:
            transcription = response.json()  # Parse the response JSON
            return transcription["text"]  # Return the transcribed text
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        # Handle any errors during the API request
        st.error(f"Error while transcribing audio: {str(e)}")
        return None

# Streamlit UI setup
st.set_page_config(page_title="Chatbot with PDF and Audio (Botify)", layout="centered")
st.title("Botify")

# Upload a PDF file
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

# Upload an audio file
audio_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a"])

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

# Handle file type and model selection
if pdf_file:
    model_choice = st.selectbox("Select the LLM model for PDF:", ["Sambanova (Qwen 2.5-72B-Instruct)", "Sambanova (Meta-Llama-3.2-1B-Instruct)"])
    st.session_state.selected_model = model_choice
    model = "Qwen2.5-72B-Instruct" if model_choice == "Sambanova (Qwen 2.5-72B-Instruct)" else "Meta-Llama-3.2-1B-Instruct"
elif audio_file:
    model_choice = "Whisper"
    st.session_state.selected_model = model_choice
    model = "whisper-1"  # Use Whisper model for audio

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

# Handle user input and process chat
user_input = st.text_input("Your message:", key="user_input", placeholder="Type your message here")
submit_button = st.button("Submit", key="submit_button")

if submit_button and user_input:
    st.session_state.current_chat.append({"role": "user", "content": user_input})

    # Process PDF content if uploaded
    if pdf_file:
        text_content = extract_text_from_pdf(pdf_file)
        prompt_text = f"Document content:\n{text_content}\n\nUser question: {user_input}\nAnswer:"
    else:
        prompt_text = f"User question: {user_input}\nAnswer:"

    # Process audio transcription if uploaded
    if audio_file:
        try:
            transcription = transcribe_audio(audio_file)
            prompt_text += f"\n\nTranscribed audio content:\n{transcription}"
        except Exception as e:
            st.error(f"Error while transcribing audio: {e}")

    st.session_state.current_chat.append({"role": "system", "content": prompt_text})

    context_length = 8192 if model == "Qwen2.5-72B-Instruct" else 16384 if model == "Meta-Llama-3.2-1B-Instruct" else 4096

    total_tokens = estimate_token_count(st.session_state.current_chat)
    if total_tokens > context_length:
        st.session_state.current_chat = st.session_state.current_chat[-3:]

    remaining_tokens = context_length - estimate_token_count(st.session_state.current_chat)
    max_tokens = min(max(remaining_tokens, 1), 1024)

    try:
        if model_choice != "Whisper":  # If not Whisper, use Sambanova API for PDF-based models
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
                st.session_state.current_chat.append({"role": "assistant", "content": answer})
                save_chat_history(st.session_state.chat_history)
            else:
                st.error("Error: Empty response from the model.")
        else:  # If Whisper model, return transcription text
            st.session_state.current_chat.append({"role": "assistant", "content": "Transcribed audio content: " + prompt_text})
            save_chat_history(st.session_state.chat_history)

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
