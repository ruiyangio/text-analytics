from graphene import ObjectType, String, Schema, Field, Float, Enum
from app.controller import get_sentiment

class Sentiment(ObjectType):
    sentiment = String()
    probability = Float()
    model = String()

class Query(ObjectType):
    sentiment = Field(Sentiment, text=String())

    def resolve_sentiment(self, info, text):
        sentiment_object = get_sentiment(text)
        return Sentiment(
            sentiment = sentiment_object["sentiment"],
            probability = sentiment_object["probability"],
            model = sentiment_object["model"]
        )

schema = Schema(query=Query)