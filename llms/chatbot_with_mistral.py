import os
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == '__main__':
    api_key = os.getenv("MISTRAL_API_KEY")
    model = "mistral-large-latest"

    # Initialize the Mistral client
    client = Mistral(api_key=api_key)

    # Define the message to send
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        }
    ]

    # Get a complete response (non-streaming)
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
    )
    print(chat_response.choices[0].message.content)

    # Get a streaming response
    stream_response = client.chat.stream(
        model=model,
        messages=messages,
    )

    for chunk in stream_response:
        print(chunk.data.choices[0].delta.content, end="")
