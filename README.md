# BandiBot
This is a Discord bot that utilizes the OpenAI API to respond to prompts in a Discord server. The bot is written in Python and uses the discord library to interact with the Discord API and the openai library to interface with the OpenAI API.

## Prerequisites
Before you can use this bot, you will need to create an API key for OpenAI. If you don't already have one, you can sign up for an API key on the OpenAI website.

In addition, you will need to have a Discord account and create a bot user for your Discord server. You can follow the instructions on the Discord Developer Portal to set up a new bot user.

## Getting started
1. Clone the repository to your local machine using the following command:

```git clone https://github.com/IanRT1/BandiBot.git```

2. Navigate to the directory where the bot is stored and install the required libraries by running the following command:

```pip install -r requirements.txt```

3. Replace the placeholder text YOUR OPENAI API KEY in line 8 with your OpenAI API key.

4. Replace the placeholder text YOUR DISCORD TOKEN in line 48 with your Discord bot user token.

5. Start the bot by running the following command:

```python BandiBot.py```

## How to use the bot
To use the bot in a Discord server, simply mention the bot in a message, followed by the prompt you would like to receive a response to. The bot will respond to the prompt in the same channel.

For example:

> @BandiBot What is the capital of France?

## Bot behavior
The bot uses the OpenAI API to generate a response to the prompt. The temperature, maximum tokens, top-p, frequency penalty, and presence penalty parameters for the API request are set to 0.5, 1024, 1, 0, and 0, respectively.

If the response from the OpenAI API contains an error, the bot will send the message An error occurred... to the Discord channel.

## Contributing
If you would like to contribute to this project, please fork the repository and create a pull request with your changes.

## Support
If you encounter any issues while using the bot, please create a new issue in the GitHub repository.
