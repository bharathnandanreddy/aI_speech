from google import genai
import os

api_key = "AIzaSyAPJAQS6OF_ul42p28PK1Cr7QjexeWun6Y"
client = genai.Client(api_key=api_key)


response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Can you give menu for a coffee shop"
)
print(response.text)