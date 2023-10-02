import aiohttp
import logging
import json
from config import OPENAI_API_KEY

# Load the config.json data
with open("config.json", "r") as file:
    config_data = json.load(file)


CATEGORY_LIST = config_data["categories"]

# Asynchronous function to categorize a message
async def categorize_message(message):
    # Create category instructions for the user
    category_instructions = "\n".join([f"{k}: {v}" for k, v in CATEGORY_LIST.items()])

    # Construct a payload for OpenAI to categorize the message
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": f"Your whole purpose is to categorize prompt intents. Use only the following categories for labeling: \n{category_instructions}\n Identify the best category for the given message and output only using the category names from the list that apply.",
            },
            {"role": "user", "content": f'Discord Message: "{message}"'},
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Send the request to OpenAI to categorize the message
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        ) as response:
            response_data = await response.json()

            # Check if 'choices' exists in the response data
            if "choices" in response_data:
                # Extract the output text and filter it to get the categories
                output_text = response_data["choices"][0]["message"]["content"].strip()
                output_categories = [cat for cat in CATEGORY_LIST if cat in output_text]
                return output_categories
            else:
                # Handle error case
                print(f"API Error: {response_data.get('error', 'Unknown error')}")
                return []


# Asynchronous function to send a payload to OpenAI
async def send_to_openai(payload):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENAI_ENDPOINT, headers=headers, json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.error(
                        f"Error from OpenAI API: {response.status} - {await response.text()}"
                    )
                    return None
    except Exception as e:
        logging.error(f"Exception during OpenAI API call: {e}")
        return None
