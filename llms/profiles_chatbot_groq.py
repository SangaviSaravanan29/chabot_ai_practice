import os
from pymongo import MongoClient
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Define the ChatMessage class if not already defined
class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}


MONGODB_URL = "mongodb+srv://hiretalent-dev:Yulwbmn87x92EQ0U@hiretalent.doscksq.mongodb.net"
DB_NAME = "app-dev"
COLLECTION_NAME = "profiles"


class ChatBot:
    def __init__(self, _api_key, model):
        self.api_key = _api_key
        self.model = model
        self.client = Groq(api_key=self.api_key)
        self.conversation_history = []
        self.initialize_context()

    def initialize_context(self):
        try:
            # Connect to MongoDB
            mongo_client = MongoClient(MONGODB_URL)
            db = mongo_client[DB_NAME]
            profiles = db[COLLECTION_NAME]

            # Fetch profile documents
            documents = profiles.find({}, {
                "_id": 0,
                "firstName": 1,
                "lastName": 1,
                "areaOfExpertise": 1,
                "careerSummary": 1,
                "type": 1
            })

            # Create a readable string from profile data
            profile_summaries = []
            for doc in documents:
                summary = (
                    f"Name: {doc.get('firstName', '')} {doc.get('lastName', '')}\n"
                    f"Slug: {doc.get('slug', '')}\n"
                    f"Expertise: {doc.get('areaOfExpertise', '')}\n"
                    f"Type: {doc.get('type', '')}\n"
                    f"Current Location: {doc.get('currentLocation', '')}\n"
                    f"Career Summary: {doc.get('careerSummary', '')}\n"
                )
                profile_summaries.append(summary)

            # Concatenate all profiles into one large system message
            context_string = "\n---\n".join(profile_summaries)
            self.conversation_history.append(
                ChatMessage(role="system", content=f"Here are all the employee profiles:\n{context_string}"))

        except Exception as e:
            print(f"Error initializing context: {e}")

    def run(self):
        print("Start chatting with the bot (type 'exit' to stop)!")
        while True:
            user_input = input("\nYou: Feel free to ask about profiles related queries: ")
            if user_input.lower() == 'exit':
                print("Exiting chat. Goodbye!")
                break

            self.conversation_history.append(ChatMessage(role="user", content=user_input))

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[msg.to_dict() for msg in self.conversation_history]
                )
                message = response.choices[0].message
                print("Bot:", message.content)
                self.conversation_history.append(ChatMessage(role=message.role, content=message.content))

            except Exception as e:
                print(f"Error during chat completion: {e}")


if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set the environment variable API_KEY")
        exit(1)

    bot = ChatBot(api_key, model="llama-3.3-70b-versatile")
    bot.run()
