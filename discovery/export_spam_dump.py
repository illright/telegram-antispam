import json
import os

from dotenv import load_dotenv
from telethon import TelegramClient, errors

load_dotenv()

chat_id = -1002296187217
api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
api_hash = os.getenv("TELEGRAM_API_HASH", "")

client = TelegramClient("anon", api_id, api_hash)


async def export_spam_dump():
    try:
        data = []
        async with client.takeout(channels=True) as takeout:
            await takeout.get_messages("me")  # wrapped through takeout (less limits)

            async for message in takeout.iter_messages(
                chat_id, wait_time=0, limit=10000
            ):
                data.append(
                    {
                        "text": message.text,
                        "date": message.date.isoformat(),
                    }
                )

            with open("message_dump.json", "w", encoding="utf-8") as f: # noqa: ASYNC230
                json.dump(data, f, ensure_ascii=False, indent=4)

    except errors.TakeoutInitDelayError as e:
        print("Must wait", e.seconds, "before takeout")


def analyze_export():
    with open("message_dump.json", "r", encoding="utf-8") as f:
        message_dump = json.load(f)
        only_text = [msg["text"] for msg in message_dump]
        import time

        start = time.time()
        print("Time taken:", time.time() - start)

        print("Total messages:", len(only_text))
        print("Total unique messages:", len(set(only_text)))


analyze_export()

# Running the code below will ask for your phone number, OTP, and your extra password.
# Then, a prompt will appear in the other Telegram apps to confirm initiating the takeout,
#   without which you would have to wait 24 hours

# with client:
#     client.loop.run_until_complete(export_spam_dump())
