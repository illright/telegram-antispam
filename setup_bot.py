import os

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

load_dotenv()


async def set_webhook():
    async with Bot(
        token=os.environ["TELEGRAM_BOT_TOKEN"],
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML,
        ),
    ) as bot:
        await bot.set_webhook(
            url=os.environ["WEBHOOK_URL"],
            allowed_updates=["message", "chat_member"],
            secret_token=os.environ["EXTRA_SECURITY_TOKEN"],
            max_connections=1,
        )


# Run with `uv run setup_bot.py`
if __name__ == "__main__":
    import asyncio

    asyncio.run(set_webhook())
