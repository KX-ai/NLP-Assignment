import os
import openai
import requests
import PyPDF2
import streamlit as st
import json
from io import BytesIO
from pydub import AudioSegment
from pydub.utils import mediainfo
from groq import Groq  # Groq library for audio transcription

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

# Updated function to preprocess and transcribe audio using the Groq API
def transcribe_audio(file):
    groq_api_key = st.secrets["groq"]["GROQ_API_KEY"]  # Access Groq API key
    client = Groq(api_key=groq_api_key)

    # Check if the file is in the correct format (MP3, WAV, etc.)
    if file.type not in ['audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/ogg', 'audio/opus', 'audio/flac']:
        st.error(f"Unsupported file type: {file.type}. Please upload an MP3, WAV, or another supported file.")
        return None

    # Check the file size (must be <= 25 MB)
    file_size = len(file.getvalue()) / (1024 * 1024)  # Convert to MB
    if file_size > 25:
        st.error(f"File is too large! The maximum allowed size is 25 MB. Your file is {file_size:.2f} MB.")
        return None

    # Use pydub to analyze the file length
    try:
        audio = AudioSegment.from_file(file)
        duration_seconds = len(audio) / 1000  # Duration in seconds
    except Exception as e:
        st.error(f"Error while reading audio file: {str(e)}")
        return None

    # Check the audio length (must be >= 10 seconds)
    if duration_seconds < 10:
        st.error(f"Audio file is too short. The minimum length is 10 seconds. Your file is {duration_seconds:.2f} seconds.")
        return None

    # Preprocess the audio to 16 kHz mono
    try:
        # Downsample the audio file to 16kHz mono
        audio = audio.set_frame_rate(16000).set_channels(1)
        processed_audio_path = "processed_audio.wav"
        audio.export(processed_audio_path, format="wav")
    except Exception as e:
        st.error(f"Error during audio preprocessing: {str(e)}")
        return None

    # Open the audio file and send it to the Groq API
    try:
        with open(processed_audio_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                file=("audio_file", audio_file),  # Send audio file
                model="whisper-large-v3-turbo",  # Use the Whisper model
                language="en",  # Specify language as English
                response_format="json",  # JSON response format
                temperature=0.0  # Optional: Set temperature (for randomness)
            )

            if transcription and 'text' in transcription:
                return transcription['text']  # Return the transcribed text
            else:
                st.error("Error: No transcription text returned.")
                return None
    except Exception as e:
        st.error(f"Error while transcribing audio with Groq: {str(e)}")
        return None

# Streamlit UI setup
st.set_page_config(page_title="Chatbot with PDF and Audio (Botify)", layout="centered")
st.title("Botify")

# Upload a PDF file
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

# Upload an audio file
audio_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a", "ogg", "opus", "flac"])

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
    model_choice = "whisper-large-v3-turbo"  # Set the Whisper model name explicitly
    st.session_state.selected_model = model_choice
    model = "whisper-large-v3-turbo"  # Ensure model is set correctly for Whisper

# Display which model is being used
st.write(f"**Model Selected:** {st.session_state.selected_model}")

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
            transcription = transcribe_audio(audio_file)  # Use the Groq transcription function
            if transcription:
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
        if model_choice != "whisper-large-v3-turbo":  # If not Whisper, use Sambanova API for PDF-based models
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
            st.session_state.current_chat.append({"role": "assistant", "content": transcription})
            save_chat_history(st.session_state.chat_history)

    except Exception as e:
        st.error(f"Error: {str(e)}")

