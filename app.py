import os
import openai
import requests
import PyPDF2
import streamlit as st
import json
from io import BytesIO
from groq import Groq  # Groq library for audio transcription

# File path for saving chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Sambanova API client class
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

# Function to extract text from PDF
@st.cache_data
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to load chat history
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    else:
        return []

# Function to save chat history
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Estimate token count for messages
def estimate_token_count(messages):
    token_count = 0
    for msg in messages:
        if "content" in msg:
            token_count += len(msg["content"].split()) * 4  # Approximation: 4 tokens/word
    return token_count

# Transcribe audio using Groq API (without ffprobe dependency)
def transcribe_audio(file):
    groq_api_key = st.secrets["groq"]["GROQ_API_KEY"]  # Use Groq API key from secrets

    # Validate file type and size
    if file.type not in ['audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/ogg', 'audio/flac']:
        st.error(f"Unsupported file type: {file.type}. Please upload a valid audio file.")
        return None

    file_size = len(file.getvalue()) / (1024 * 1024)
    if file_size > 25:
        st.error(f"File is too large! Max size is 25 MB. Your file is {file_size:.2f} MB.")
        return None

    # Create a BytesIO buffer from the uploaded audio file
    buffer = BytesIO(file.getvalue())
    buffer.name = file.name  # Set the file name to help with format detection
    buffer.seek(0)  # Reset buffer position

    # Transcribe audio using Whisper
    try:
        transcription = openai.Audio.transcribe(
            model="whisper-1",  # Use Whisper model for transcription
            file=("audio_file", buffer)
        )
        return transcription.get('text', "No transcription text returned.")
    except Exception as e:
        st.error(f"Error while transcribing audio: {str(e)}")
        return None

# Streamlit app setup
st.set_page_config(page_title="Botify Chatbot", layout="centered")
st.title("Botify: PDF and Audio Assistant")

# Upload a PDF file
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

# Upload an audio file
audio_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! I am Botify, your assistant. How can I assist you today?"}]
    st.session_state.selected_model = "Qwen2.5-72B-Instruct"

# Start a new chat
if st.button("Start New Chat"):
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! Starting a new conversation."}]
    st.session_state.chat_history.append(st.session_state.current_chat)
    st.success("New chat started!")

# Handle file uploads
if pdf_file:
    st.session_state.selected_model = "Qwen2.5-72B-Instruct"
    pdf_text = extract_text_from_pdf(pdf_file)
    st.write(f"**PDF Content Extracted:** {pdf_text[:500]}...")

if audio_file:
    st.session_state.selected_model = "whisper-1"  # Updated model for Groq API
    transcription = transcribe_audio(audio_file)
    if transcription:
        st.write(f"**Audio Transcription:** {transcription}")

# Chat interface
st.write("### Chat Conversation")
for msg in st.session_state.current_chat:
    if msg["role"] == "user":
        st.markdown(f"**User:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Botify:** {msg['content']}")

# User input
user_input = st.text_input("Your message:", placeholder="Ask a question or say something...")
if st.button("Submit"):
    st.session_state.current_chat.append({"role": "user", "content": user_input})

    # Create prompt
    prompt = f"User: {user_input}\nAssistant:"
    if pdf_file:
        prompt = f"PDF Content: {pdf_text}\n\n{prompt}"
    if audio_file:
        prompt += f"\n\nAudio Transcription: {transcription}"

    # Process response
    sambanova_api_key = st.secrets["general"]["SAMBANOVA_API_KEY"]
    model = st.session_state.selected_model
    client = SambanovaClient(api_key=sambanova_api_key, base_url="https://api.sambanova.ai/v1")

    try:
        response = client.chat(
            model=model,
            messages=st.session_state.current_chat,
            temperature=0.7,
            top_p=1.0,
            max_tokens=500
        )
        answer = response["choices"][0]["message"]["content"]
        st.session_state.current_chat.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.error(f"Error: {str(e)}")
