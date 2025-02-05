# Antispam Telegram bot

This bot fights Russian spam in Telegram chats.

Features:

- Built to be self-hosted
- Minimalistic so that it's easy to understand the source code
- Only works in chats that it recognizes (configurable)
- Fights reaction spammers too (users who join, react to a lot of messages, then leave; they usually have spam in their bio)
- Can forward all the spam to the channel for additional manual review (known as a _spam dump_)

## How to self-host for free

This is an AI-powered bot, so ethics are crucial:

1. This bot **only bans users for a short period of time** because it's not perfect.  
   Spammers don't come back, and if they do, they will be banned every time.
2. If you have resources, **monitor the spam dump for wrongly banned users**.  
   Getting banned by a bot without knowing why _really sucks_.
3. It's powered by a small classifier to be **resource-efficient**, not 100% accurate.  
   The added benefit is that you can host it on Modal and barely use any monthly credits.
4. The classifier is **strictly for non-commercial purposes**.  
   The author of the classifier, [NeuroSpaceX](https://huggingface.co/NeuroSpaceX), chose the CC-BY-NC-4.0 license. Respect that.

<details><summary>I understand, give me the instructions</summary>

&nbsp;  
You will need [uv](https://github.com/astral-sh/uv) installed.

1. Clone this project
2. Run `uv sync` to install dependencies
3. Rename the `.env.sample` file to `.env`
4. Create an account on [modal.com](https://modal.com); that's where the bot will be hosted
   1. Run `uv run modal setup` to log in from the terminal
5. Download the `classifier.zip` archive from [the latest release on GitHub](https://github.com/illright/telegram-antispam/releases/latest) (look for the section "Assets") and unpack it here in this folder
6. Create a Telegram bot using [@BotFather](https://t.me/BotFather)
   1. Copy the bot token into `TELEGRAM_BOT_TOKEN` in the `.env` file, removing the sample value
7. [Create a long password](https://bitwarden.com/password-generator/#password-generator) (50 characters) and copy it into `EXTRA_SECURITY_TOKEN` in the `.env` file, removing the sample value
8. Create a secret on Modal with these commands:
   1. `source .env` to load the environment variables into your shell
   2. `uv run modal secret create antispam-telegram-bot-token EXTRA_SECURITY_TOKEN=$EXTRA_SECURITY_TOKEN TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN`
9. Create a channel where the bot can forward spam for manual review. Invite your Telegram bot as a member (Optional, but recommended).
10. Go to `main.py` and find `allowed_chats`. Replace the sample chat with your own. See the descriptions of parameters in the `ChatSettings` class. All parameters can be omitted.
11. Run `uv run modal deploy main` to deploy the bot to Modal
    1. It will print the web endpoint:
       ```
       ├── 🔨 Created web endpoint for Model.process_update => https://something-something.modal.run
       ```
       Copy `https://something-something.modal.run` into `WEBHOOK_URL` in the `.env` file, removing the sample value
12. Run `uv run setup_bot.py` to connect the bot to your Modal app and set the extra security token
13. Invite the bot to your chat. Done!

</details>

## Navigate the source code

The code of the bot is almost entirely contained in [`main.py`](./main.py). The logic of what the bot does is inside the `setup_bot` method in the `Model` class.

The `clean_text` folder contains the pre-processing functions for the messages. Only `v7_tiny.py` is used by the bot.

The `discovery` folder is not used by the bot, it contains code that was used earlier to run experiments, preserved for reference.

The `setup_bot.py` is a script that is used to connect the bot to the Modal app and set the extra security token.

## License

- The source code of the bot is licensed under GNU AGPL-3.0.  
  Explained: https://choosealicense.com/licenses/agpl-3.0

- The spam classifier that is used by the bot, is licensed by [NeuroSpaceX](https://huggingface.co/NeuroSpaceX) under CC-BY-NC-4.0.  
  Explained: https://creativecommons.org/licenses/by-nc/4.0/
