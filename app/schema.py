from graphene import ObjectType, String, Schema, Field, Float, Enum
from app.controller import get_sentiment

class Sentiment(ObjectType):
    sentiment = String()
    probability = Float()

class Query(ObjectType):
    sentiment = Field(Sentiment, text=String())

    def resolve_sentiment(self, info, text):
        sentimentObject = get_sentiment(text)
        return Sentiment(
            sentiment = sentimentObject["sentiment"],
            probability = sentimentObject["probability"]
        )

schema = Schema(query=Query)