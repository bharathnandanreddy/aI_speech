
import json
from google import genai

def get_menu(filename="menu.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            menu = json.load(f)
    except Exception as e:
        print(f"Error loading menu from {filename}: {e}\nUsing default menu.")
        menu = {
            "Coffee": 3.0,
            "Espresso": 2.5,
            "Cappuccino": 3.5,
            "Latte": 4.0,
            "Tea": 2.0,
            "Muffin": 2.5,
            "Croissant": 3.0
        }
    return menu

api_key = "AIzaSyAPJAQS6OF_ul42p28PK1Cr7QjexeWun6Y"
client = genai.Client(api_key=api_key)
menu = get_menu()
menu_str = "\n".join([f"{item}: ${price:.2f}" for item, price in menu.items()])
cart = {}


# Use a single 'contents' list with 'parts' and 'role' for Gemini SDK
contents = [
    {
        "role": "user",
        "parts": [
            {"text": f"New Session started! You are a friendly coffee shop barista. Here is the menu:\n{menu_str}\nGreet the new customer when they say hello and ask for their order. When they order, confirm the item and quantity, and keep a running cart. When the customer is done, immediately summarize the order, return the total price, and end the conversation by saying 'Thank you for your order!'. Only offer items from the menu. Do not ask for more orders after the user is done."}
        ]
    }
]
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents
)
contents.append({
    "role": "model",
    "parts": [{"text": response.text}]
})
while True:
    user_input = input("You: ")
    contents.append({
        "role": "user",
        "parts": [{"text": user_input}]
    })

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )
    contents.append({
        "role": "model",
        "parts": [{"text": response.text}]
    })

    print("Barista:", response.text)
    if "Thank you for your order" in response.text:  
        contents.clear()
        contents.append({
            "role": "user",
            "parts": [
                {"text": f"New Session started! You are a friendly coffee shop barista. Here is the menu:\n{menu_str}\nGreet the new customer when they say hello and ask for their order. When they order, confirm the item and quantity, and keep a running cart. When the customer is done, immediately summarize the order, return the total price, and end the conversation by saying 'Thank you for your order!'. Only offer items from the menu. Do not ask for more orders after the user is done."}
            ]
        })
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        contents.append({
            "role": "model",
            "parts": [{"text": response.text}]
        })

