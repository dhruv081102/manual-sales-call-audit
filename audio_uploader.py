import streamlit as st
import requests
import json
from pymongo import MongoClient
import openai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Initialize the OpenAI client with your API key
openai_api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=openai_api_key)

# MongoDB Atlas configuration
mongodb_uri = "mongodb+srv://dhruv:dhruv1234@cluster0.fhkve.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client_mongo = MongoClient(mongodb_uri)
db = client_mongo.rosewalt_db
collection = db.call_analysis

# Function to transcribe audio files using OpenAI's Whisper API
def transcribe_audio(file):
    url = 'https://api.openai.com/v1/audio/transcriptions'
    
    if not file.name.lower().endswith(('.wav', '.mp3', '.m4a')):
        raise ValueError(f"File {file.name} is not an audio file.")

    files = {
        'file': (file.name, file, 'application/octet-stream')
    }
    headers = {
        'Authorization': f'Bearer {client.api_key}'
    }
    data = {
        'model': 'whisper-1',
        'language': 'en'
    }

    response = requests.post(url, headers=headers, data=data, files=files)
    
    if response.status_code == 200:
        result = response.json()
        text = result.get('text', '')
        duration = result.get('duration', estimate_duration_from_transcription(text))
        return {"text": text, "duration": duration}
    else:
        error_message = response.json().get('error', {}).get('message', 'Unknown error')
        raise Exception(f"Error transcribing audio file: {error_message}")

def estimate_duration_from_transcription(transcription_text):
    if not transcription_text:
        return None
    
    average_words_per_minute = 120  
    word_count = len(transcription_text.split())
    duration_in_seconds = round(word_count / (average_words_per_minute / 60), 2)
    
    return duration_in_seconds

def evaluate_transcription_with_openai(text):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "call_audit",
                "description": "You are a function who will do the audit on the calls",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pitch_followed": {"type": "integer", "description": "This measures how well the sales representative understands and articulates the details of the flats being offered."},
                        "confidence": {"type": "integer", "description": "This reflects how well the sales agent is able to address and clarify the doubts of the prospect."},
                        "tonality": {"type": "integer", "description": "This includes the tone in which the sales representative is speaking and how well they manage the tone throughout the conversation."},
                        "energy": {"type": "integer", "description": "This measures the level of enthusiasm and energy the sales representative exhibits during the call."},
                        "enthusiasm": {"type": "integer", "description": "This reflects the sales representative's eagerness and interest in discussing the flats."},
                        "customer_understanding": {"type": "integer", "description": "This reflects the representative's ability to understand the customer's needs, problems, and concerns."},
                        "communication_skills": {"type": "integer", "description": "This includes the clarity, tone, pace, use of persuasive language, and active listening."},
                        "objection_handling": {"type": "integer", "description": "This measures the representative's ability to address customer objections effectively."},
                        "closing_skills": {"type": "integer", "description": "This measures the representative's ability to guide the conversation towards a sale or next step."},
                        "Overall Score": {"type": "decimal", "description": "Your ultimate goal is to evaluate all the scores and give them an overall score."},
                        "conclusion": {"type": "string", "description": "Your ultimate goal is to evaluate the salesperson's performance and give them points for improvement."}
                    },
                    "additionalProperties": False,
                    "required": [
                        "pitch_followed",
                        "confidence",
                        "tonality",
                        "energy",
                        "enthusiasm",
                        "customer_understanding",
                        "communication_skills",
                        "objection_handling",
                        "closing_skills",
                        "Overall Score",
                        "conclusion"
                    ]
                },
                "strict": True
            }
        }
    ]

    tool_prompt = f"""
    tools={json.dumps(tools, indent=4)},
    response_format={{
        "type": "text"
    }}
    """

    prompt = f"""
    Based on the following transcription of a sales call, perform the call audit using the provided tool.

    For each of the following aspects:
    - pitch_followed
    - confidence
    - tonality
    - energy
    - enthusiasm
    - customer_understanding
    - communication_skills
    - objection_handling
    - closing_skills

   Make sure that you also give a detailed explanation of each metric with verbatim examples of what was said to support your explanation.
Your ultimate goal is to evaluate the salesperson's performance and give them points for improvement by citing verbatim examples of what was said during the conversation. Then give an overall score. Be highly critical
    Transcription:
    "{text}"

    Use the provided tool and output the evaluation as JSON.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": tool_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    message_content = response.choices[0].message.content

    evaluation = json.loads(message_content)
    return evaluation

def save_to_mongodb(file_name, transcription_result, evaluation_result, salesperson_name, prospect_name):
    data_to_save = {
        "file_name": file_name,
        "salesperson_name": salesperson_name,
        "prospect_name": prospect_name,
        "transcription": transcription_result['text'],
        "estimated_duration": f"{transcription_result['duration']} seconds",
        "evaluation": evaluation_result
    }
    collection.insert_one(data_to_save)
    st.success(f"Transcription and evaluation for '{file_name}' saved to MongoDB.")

# Streamlit UI Setup
st.title("Audio File Uploader, Transcriber, Evaluator, and Dashboard")

# Instructions
st.write("Please upload your audio files (WAV, MP3, M4A). The files will be transcribed using OpenAI's Whisper API, and the transcription will be evaluated on various factors.")

# File uploader widget for multiple files
uploaded_files = st.file_uploader("Choose audio files", type=["wav", "mp3", "m4a"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing file: {uploaded_file.name}")

        try:
            transcription_result = transcribe_audio(uploaded_file)

            if transcription_result['duration'] > 200:
                salesperson_name = st.text_input(f"Enter the Salesperson's Name for {uploaded_file.name}")
                prospect_name = st.text_input(f"Enter the Prospect's Name for {uploaded_file.name}")

                if salesperson_name and prospect_name:
                    evaluation_result = evaluate_transcription_with_openai(transcription_result['text'])

                    st.success(f"Transcription for '{uploaded_file.name}':")
                    st.write(transcription_result['text'])
                    st.write(f"Estimated Duration: {transcription_result['duration']} seconds")

                    st.success(f"Evaluation for '{uploaded_file.name}':")
                    st.json(evaluation_result)

                    save_to_mongodb(uploaded_file.name, transcription_result, evaluation_result, salesperson_name, prospect_name)
                else:
                    st.warning(f"Please enter both the Salesperson's Name and the Prospect's Name to proceed for {uploaded_file.name}.")
            else:
                st.warning(f"The call duration for '{uploaded_file.name}' is {transcription_result['duration']} seconds, which is less than the required 200 seconds.")
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")

# Dashboard Section
st.subheader("Search and View Results")

search_criteria = st.selectbox("Search by:", ["Salesperson's Name", "Prospect's Name", "File Name"])

search_query = st.text_input(f"Enter the {search_criteria}")

if search_query:
    search_field = {
        "Salesperson's Name": "salesperson_name",
        "Prospect's Name": "prospect_name",
        "File Name": "file_name"
    }.get(search_criteria)

    # Query MongoDB
    results = collection.find({search_field: {"$regex": search_query, "$options": "i"}})

    # Convert the cursor to a list to handle it
    results_list = list(results)

    if results_list:
        st.write("Results:")
        for result in results_list:
            st.write(f"*File Name*: {result.get('file_name')}")
            st.write(f"*Salesperson's Name*: {result.get('salesperson_name')}")
            st.write(f"*Prospect's Name*: {result.get('prospect_name')}")
            st.write(f"*Transcription*: {result.get('transcription')}")
            st.write(f"*Estimated Duration*: {result.get('estimated_duration')}")
            st.write(f"*Evaluation*:")
            st.json(result.get('evaluation'))
            st.write("---")
    else:
        st.write("No results found.")