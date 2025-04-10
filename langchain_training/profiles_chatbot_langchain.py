import os
import getpass
from pymongo import MongoClient
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class ChatBot:
    def __init__(self, _api_key, model_name):
        self.api_key = _api_key
        self.model = init_chat_model(model_name, model_provider="mistralai")
        self.conversation_history = []
        self.context = ""
        self.initialize_context()

    def initialize_context(self):
        try:
            # Retrieve MongoDB connection details from environment variables
            mongodb_url = os.getenv("MONGODB_URL")
            db_name = os.getenv("DB_NAME")
            collection_name = os.getenv("COLLECTION_NAME")

            # Connect to MongoDB
            mongo_client = MongoClient(mongodb_url)
            db = mongo_client[db_name]
            profiles = db[collection_name]

            # Fetch profile documents
            documents = profiles.find({}, {
                "_id": 0,
                "firstName": 1,
                "lastName": 1,
                "areaOfExpertise": 1,
                "careerSummary": 1,
                "type": 1,
                "currentLocation": 1
            })

            # Create a readable string from profile data
            profile_summaries = []
            for doc in documents:
                summary = (
                    f"Name: {doc.get('firstName', '')} {doc.get('lastName', '')}\n"
                    f"Expertise: {doc.get('areaOfExpertise', '')}\n"
                    f"Type: {doc.get('type', '')}\n"
                    f"Location: {doc.get('currentLocation', '')}\n"
                    f"Career Summary: {doc.get('careerSummary', '')}\n"
                )
                profile_summaries.append(summary)

            # Concatenate all profiles into one large context string
            self.context = "\n---\n".join(profile_summaries)

        except Exception as e:
            print(f"Error initializing context: {e}")

    def create_prompt(self, context, user_input):
        system_template = (
            "You are a helpful assistant providing information about employee profiles. "
            "Here are the profiles:\n{context}\n\n"
            "User: {user_input}\nAssistant:"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "{user_input}")
        ])

        # Invoke the prompt template with the provided context and user input
        messages = prompt_template.format_messages(
            context=context,
            user_input=user_input
        )

        return messages

    def run(self):
        print("Start chatting with the bot (type 'exit' to stop)!")
        while True:
            user_input = input("\nYou: Feel free to ask about profiles related queries: ")
            if user_input.lower() == 'exit':
                print("Exiting chat. Goodbye!")
                break

            self.conversation_history.append(ChatMessage(role="user", content=user_input))

            try:
                # Prepare the prompt with context and user input
                messages = self.create_prompt(self.context, user_input)

                # Get the response from the model
                response = self.model.invoke(messages)
                print("Bot:", response.content)
                self.conversation_history.append(ChatMessage(role="assistant", content=response.content))

            except Exception as e:
                print(f"Error during chat completion: {e}")


if __name__ == "__main__":
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter API key for Mistral AI: ")

    bot = ChatBot(api_key, model_name="mistral-large-latest")
    bot.run()
