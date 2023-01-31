# BandiBot

This is a simple Discord bot that uses the discord.py library and OpenAI API to respond to user prompts in a Discord server.

Requirements

discord.py library (version 1.5.1)
openai library (version 0.7.0)
OpenAI API Key
To install the requirements, run:

pip install -r requirements.txt

Running the Bot

Create a new Discord client using discord.py library
Set up the OpenAI client using the API Key
Start the bot using the TOKEN environment variable
Features
The bot will respond to every prompt in a Discord server as accurately as possible, using OpenAI API to generate responses. If it doesn't know the answer, it will create a fake one. The bot ignores messages from itself and only responds to messages that mention the bot.

Contributing
If you'd like to contribute to the project, please fork the repository and make your changes. Submit a pull request after testing your changes and I'll review and merge it.
