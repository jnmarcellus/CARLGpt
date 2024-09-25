import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
import pyperclip

# Page configuration settings for the Streamlit app
PAGE_CONFIG = {
    "page_title": "CARL", 
    "layout": "centered", 
    "initial_sidebar_state": "auto"
}

# Set the page configuration using the defined settings
st.set_page_config(**PAGE_CONFIG)

# Configure logging to display information during runtime
logging.basicConfig(level=logging.INFO)

# Initialize chat history in session state if it does not exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

def stream_chat(model, messages):
    """
    Stream the chat response from the selected model.

    Args:
        model (str): The name of the model to use for generating responses.
        messages (list): A list of messages in the chat history.

    Returns:
        str: The concatenated response from the model.

    Raises:
        Exception: If an error occurs during the streaming process.
    """
    try:
        # Initialize the model with a request timeout
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)  # Start streaming responses from the model
        response = ""
        response_placeholder = st.empty()  # Placeholder for dynamic response display
        
        # Append streamed response segments to the response variable
        for r in resp:
            response += r.delta
            response_placeholder.write(response)  # Update the placeholder with the current response
        
        # Log the model used and the messages exchanged
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        # Log any errors encountered during the streaming process
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def copy_to_clipboard(message):
    """
    Copy the given message content to the clipboard.

    Args:
        message (str): The message content to be copied to clipboard.

    Returns:
        None
    """
    pyperclip.copy(message)  # Copy the message to the clipboard
    st.success("Message copied to clipboard!")  # Notify the user

def main():
    """
    Main function to run the Streamlit app.

    This function sets up the user interface, handles user inputs, and
    orchestrates the chat functionality.
    """
    # Set the main title and subtitle for the app
    st.markdown("<h1 style='text-align: center;'>CARL (Research)</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Corporate Assistant for Rapid Lookups</h2>", unsafe_allow_html=True)
    logging.info("App started")

    # Sidebar for model selection
    model = st.sidebar.selectbox("Choose a model", ["llama3.2:1b","llama3.1", "tinyllama", "llama3", "mistral-small"])
    logging.info(f"Model selected: {model}")

    # Sidebar option to display duration of response
    show_duration = st.sidebar.radio("Show Duration?", ["Yes", "No"])

    # Button to clear chat history
    if st.sidebar.button("Clear History"):
        st.session_state.messages.clear()  # Clear the chat history
        st.success("Chat history cleared!")  # Notify the user

    # Input field for user question
    prompt = st.chat_input("Your question")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})  # Append user prompt to chat history
        logging.info(f"User input: {prompt}")

        # Display user messages from the chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])  # Display the message content

            # Create a button to copy user message to clipboard
            if message["role"] == "user":
                if st.button("🔗", key=f"user_{message['content']}", help="Copy this message to clipboard", 
                             on_click=copy_to_clipboard, args=(message["content"],)):
                    pass  # Handle button click (no additional actions)

        # Generate a response only if the last message was from the user
        if st.session_state.messages[-1]["role"] == "user":
            start_time = time.time()  # Start timing the response generation
            logging.info("Generating response")

            with st.spinner("Writing... Remember this is running on John's Laptop"):
                try:
                    # Prepare messages for the model
                    messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                    response_message = stream_chat(model, messages)  # Get the model's response
                    duration = time.time() - start_time  # Calculate response duration

                    # Format the response to include duration if selected
                    if show_duration == "Yes":
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                    else:
                        response_message_with_duration = response_message

                    # Append the assistant's response to the chat history if it's not a duplicate
                    if not st.session_state.messages or st.session_state.messages[-1]["role"] != "assistant":
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})

                    # Display the assistant's response
                    with st.chat_message("assistant"):
                        st.write(response_message_with_duration)

                    # Create a button to copy assistant's response to clipboard
                    if st.button("🔗", key=f"assistant_{response_message_with_duration}", help="Copy this message to clipboard", 
                                 on_click=copy_to_clipboard, args=(response_message_with_duration,)):
                        pass  # Handle button click (no additional actions)

                    logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                except Exception as e:
                    # Handle errors during response generation
                    st.session_state.messages.append({"role": "assistant", "content": str(e)})
                    st.error("An error occurred while generating the response.")
                    logging.error(f"Error: {str(e)}")

# Entry point for running the app
if __name__ == "__main__":
    main()
