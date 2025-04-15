import os
from enum import Enum
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


class SentimentEnum(str, Enum):
    happy = "happy"
    neutral = "neutral"
    sad = "sad"


class Person(BaseModel):
    """Information about a person."""
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(default=None, description="The color of the person's hair if known")
    height_in_meters: Optional[str] = Field(default=None, description="Height measured in meters")


class FullAnalysis(BaseModel):
    """Full analysis including sentiment, language, aggression, and person info."""
    sentiment: SentimentEnum = Field(description="The sentiment of the text")
    aggressiveness: int = Field(description="Aggressiveness level on a scale from 1 to 10")
    language: str = Field(description="The language the text is written in")
    people: List[Person] = Field(default_factory=list)


class TextAnalyzer:
    def __init__(self, model_name: str = "mistral-large-latest", provider: str = "mistralai"):
        """Initialize the analyzer with the model and structured output."""
        self.llm = init_chat_model(model_name, model_provider=provider)
        self.llm_with_output = self.llm.with_structured_output(FullAnalysis)

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert extraction algorithm. "
                    "Only extract relevant information from the text. "
                    "If you do not know the value of an attribute, return null.",
                ),
                ("human", "{text}"),
            ]
        )

    def analyze(self, text: str) -> FullAnalysis:
        """Analyze the input text and return structured output."""
        prompt = self.prompt_template.invoke({"text": text})
        result = self.llm_with_output.invoke(prompt)

        if not isinstance(result, FullAnalysis):
            raise TypeError(f"Expected FullAnalysis, got {type(result).__name__}")
        return result

    def analyze_as_dict(self, text: str) -> Dict[str, Any]:
        """Return analysis as dictionary."""
        return self.analyze(text).model_dump()


def main():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Please set the environment variable MISTRAL_API_KEY")
        return

    analyzer = TextAnalyzer()

    text = (
        "My name is Jeff, my hair is black and I am 6 feet tall. "
        "Anna has the same color hair as me. I feel very happy today."
    )

    results = analyzer.analyze_as_dict(text)
    print(results)


if __name__ == "__main__":
    main()
