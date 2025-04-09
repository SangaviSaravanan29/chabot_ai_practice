import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file
load_dotenv()

if __name__ == '__main__':
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = "deepseek/deepseek-r1:free"

    # Initialize the OpenRouter client with correct base URL
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Define the message to send
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        }
    ]

    # Get a complete response (non-streaming)
    chat_response = client.chat.completions.create(model=model, messages=messages)  # Updated method
    print("Complete response:")
    print(chat_response.choices[0].message.content)  # Updated attribute access

    # Get a streaming response
    print("\nStreaming response:")
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
