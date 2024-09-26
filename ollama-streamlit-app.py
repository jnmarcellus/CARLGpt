import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import logging
import pyperclip
import time
import ollama
from typing import Dict, Generator

def ollama_generator(model_name: str, messages: Dict) -> Generator:
    stream = ollama.chat(
        model=model_name, messages=messages, stream=True)
    for chunk in stream:
        yield chunk['message']['content']

# Page configuration settings for the Streamlit app
PAGE_CONFIG = {
    "page_title": "CARL", 
    "layout": "centered", 
    "initial_sidebar_state": "auto",
    "page_icon": "🦜",
    "initial_sidebar_state": "collapsed"
}

# Set the page configuration using the defined settings
st.set_page_config(**PAGE_CONFIG)

# Configure logging to display information during runtime
logging.basicConfig(level=logging.INFO)

# Initialize chat history in session state if it does not exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

def stream_chat(model, messages):
    try:
        # Initialize the model with a request timeout
        llm = Ollama(model=model, request_timeout=240.0)
        resp = llm.stream_chat(messages)  # Start streaming responses from the model
        response = ""
        response_placeholder = st.empty()  # Placeholder for dynamic response display
        
        # Append streamed response segments to the response variable
        for r in resp:
            response += r.delta
            response_placeholder.write(response)  # Update the placeholder with the current response
            
            # Append the current segment to the session state as an assistant message
            st.session_state.messages.append({"role": "assistant", "content": r.delta})
        
        # Log the model used and the messages exchanged
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        # Log any errors encountered during the streaming process
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def copy_to_clipboard(message):
    pyperclip.copy(message)  # Copy the message to the clipboard
    st.success("Message copied to clipboard!")  # Notify the user

def main():
    # Set the main title and subtitle for the app
    st.markdown("<h1 style='text-align: center;'>CARL (Research)</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Corporate Assistant for Rapid Lookups</h2>", unsafe_allow_html=True)
    logging.info("App started")

    # Sidebar for model selection
    model = st.sidebar.selectbox("Choose a model", ["llama3.2:1b", "llama3.1", "tinyllama", "llama3", "mistral-small"])
    logging.info(f"Model selected: {model}")
    

    # Sidebar option to display duration of response
    show_duration = st.sidebar.radio("Show Duration?", ["Yes", "No"])
    logging.info(f"Show Duration: {show_duration}")

    # Button to clear chat history
    if st.sidebar.button("Clear Cache"):
        st.cache_data.clear()  # Clear the cache
        success_message = st.success("Cache cleared!")  # Show success message
        
        # Use a placeholder to manage the message display
        time.sleep(5)  # Wait for 5 seconds
        success_message.empty()  # Remove the success message

    # Button to clear data cache
    if st.sidebar.button("Clear History"):
        st.session_state.messages.clear()  # Clear the chat history
        success_message = st.success("Chat History cleared!")  # Show success message
        
        # Use a placeholder to manage the message display
        time.sleep(5)  # Wait for 5 seconds
        success_message.empty()  # Remove the success message

    # Input field for user question
    prompt = st.chat_input("Your question")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})  # Append user prompt to chat history
        logging.info(f"User input: {prompt}")

        # Display user messages from the chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])  # Display the message content

        # Generate a response only if the last message was from the user
        if st.session_state.messages[-1]["role"] == "user":
            start_time = time.time()  # Start timing the response generation
            logging.info("Generating response")

        with st.spinner("Thinking..."):
            try:
                # Prepare messages for the model
                messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                
                # Get the model's response
                response_message = stream_chat(model, messages)

                # Log the response and duration
                duration = time.time() - start_time
                logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                # Append the response to the session state as the assistant's message
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_message})

                #with st.chat_message("assistant"):
                #    response = st.write_stream(ollama_generator(
                #        st.session_state.selected_model, st.session_state.messages))
                #st.session_state.messages.append(
                #    {"role": "assistant", "content": response})


            except Exception as e:
                # Handle errors during response generation
                error_message = "I am sorry Dave, I cannot do that."  # More user-friendly error message
                st.session_state.messages.append({"role": "assistant", "content": error_message})  # Append error as assistant message
                st.error(error_message)
                logging.error(f"Error: {str(e)}")



# Entry point for running the app
if __name__ == "__main__":
    main()
