from graphene import ObjectType, String, Schema, Field, Float, Enum
from app.controller import get_sentiment

class Sentiment(ObjectType):
    sentiment = String()
    logProbability = Float()

class Query(ObjectType):
    sentiment = Field(Sentiment, text=String())

    def resolve_sentiment(self, info, text):
        sentimentObject = get_sentiment(text)
        return Sentiment(
            sentiment = sentimentObject["sentiment"],
            logProbability = sentimentObject["logProbability"]
        )

schema = Schema(query=Query)