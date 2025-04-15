import os
from enum import Enum
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()


class SentimentEnum(str, Enum):
    happy = "happy"
    neutral = "neutral"
    sad = "sad"


class Classification(BaseModel):
    sentiment: SentimentEnum = Field(description="The sentiment of the text")
    aggressiveness: int = Field(description="Aggressiveness level on a scale from 1 to 10")
    language: str = Field(description="The language the text is written in")


class SentimentAnalyzer:
    def __init__(self, model_name: str = "mistral-large-latest", provider: str = "mistralai"):
        """Initialize the sentiment analyzer with the specified LLM."""
        self.llm = init_chat_model(model_name, model_provider=provider)
        self.llm_with_output = self.llm.with_structured_output(Classification)
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            Extract the desired information from the following passage.
            Only extract the properties mentioned in the 'Classification' function.
            Passage:
            {input}
            """
        )

    def analyze_text(self, text: str) -> Classification:
        """Analyze the sentiment, aggressiveness, and language of the input text."""
        prompt = self.prompt_template.invoke({"input": text})
        response = self.llm_with_output.invoke(prompt)

        # Ensure the response is of type Classification
        if not isinstance(response, Classification):
            raise TypeError(f"Expected type 'Classification', got '{type(response).__name__}' instead")

        return response

    def analyze_to_dict(self, text: str) -> Dict[str, Any]:
        """Analyze the text and return results as a dictionary."""
        response = self.analyze_text(text)
        return response.model_dump()


def main():
    # Get Mistral AI API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Please set the environment variable MISTRAL_API_KEY")
        return
    # Create analyzer instance
    analyzer = SentimentAnalyzer()
    # Sample text
    text = "உங்களையே நம்புங்கள்"
    # Analyze text and print results
    results = analyzer.analyze_to_dict(text)
    print(results)


if __name__ == "__main__":
    main()
