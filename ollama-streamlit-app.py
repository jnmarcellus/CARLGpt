import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
import pyperclip

PAGE_CONFIG = {
    "page_title": "CARL", 
    "layout": "centered", 
    "initial_sidebar_state": "auto"
}

st.set_page_config(**PAGE_CONFIG)

logging.basicConfig(level=logging.INFO)

# Initialize chat history in session state if not already present
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to stream chat response based on selected model
def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0) 
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def copy_to_clipboard(message):
    """Copy the message content to the clipboard."""
    pyperclip.copy(message)
    st.success("Message copied to clipboard!")

def main():
    st.markdown("<h1 style='text-align: center;'>CARL (Research)</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Corporate Assistant for Rapid Lookups</h2>", unsafe_allow_html=True)
    logging.info("App started")

    # Sidebar for model selection
    model = st.sidebar.selectbox("Choose a model", ["llama3.2:1b","llama3.1", "tinyllama", "llama3", "mistral-small",])
    logging.info(f"Model selected: {model}")

    # Sidebar for showing duration
    show_duration = st.sidebar.radio("Show Duration?", ["Yes", "No"])

    # Sidebar button to clear chat history
    if st.sidebar.button("Clear History"):
        st.session_state.messages.clear()  # Clear the chat history
        st.success("Chat history cleared!")  # Feedback to the user

    # Input for user question
    prompt = st.chat_input("Your question")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        # Display user messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

            # Create a copy button next to the user message
            if message["role"] == "user":
                if st.button("🔗", key=f"user_{message['content']}", help="Copy this message to clipboard", 
                             on_click=copy_to_clipboard, args=(message["content"],)):
                    pass  # Just to handle the button click without any additional actions

        # Only generate a response if the last message is from the user
        if st.session_state.messages[-1]["role"] == "user":
            start_time = time.time()
            logging.info("Generating response")

            with st.spinner("Writing... Remember this is running on John's Laptop"):
                try:
                    messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                    response_message = stream_chat(model, messages)
                    duration = time.time() - start_time

                    # Prepare the final response message
                    if show_duration == "Yes":
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                    else:
                        response_message_with_duration = response_message

                    # Only append the response if it's not already present
                    if not st.session_state.messages or st.session_state.messages[-1]["role"] != "assistant":
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})

                    # Display assistant's response
                    with st.chat_message("assistant"):
                        st.write(response_message_with_duration)

                    # Create a copy button next to the assistant message
                    if st.button("🔗", key=f"assistant_{response_message_with_duration}", help="Copy this message to clipboard", 
                                 on_click=copy_to_clipboard, args=(response_message_with_duration,)):
                        pass  # Just to handle the button click without any additional actions

                    logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": str(e)})
                    st.error("An error occurred while generating the response.")
                    logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
