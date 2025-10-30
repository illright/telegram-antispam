import hmac
import io
import logging
import os
import re
from datetime import timedelta
from typing import Annotated, List
from dataclasses import dataclass, field

import modal
from fastapi import Header
from aiogram import types

from clean_text.v7_tiny import clean_text

classifier_path = "./ruSpamNS_v7_tiny"
if os.path.exists(os.path.join("classifier", classifier_path)):
    # If the archive extractor created an extra folder "classifier", go along with it
    classifier_path = os.path.normpath(os.path.join("classifier", classifier_path))

remote_project_path = "/app"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("tesseract-ocr", "tesseract-ocr-rus")
    .uv_sync()
    .add_local_dir(
        classifier_path,
        remote_path=os.path.join(remote_project_path, classifier_path),
        copy=True,
    )
    .add_local_python_source("clean_text")
)

with image.imports():
    import torch
    import pytesseract
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from aiogram import Bot, Dispatcher, types
    from aiogram.enums import ParseMode
    from aiogram.client.default import DefaultBotProperties
    from aiogram.filters import IS_MEMBER, IS_NOT_MEMBER, ChatMemberUpdatedFilter
    from PIL import Image


app = modal.App(name="antispam", image=image)
unchecked_users = modal.Dict.from_name("unchecked_users", create_if_missing=True)


@dataclass
class ChatSettings:
    """Preferences for how the bot should work in a particular chat."""

    # If True, the bot will only notify admins about spam, but won't ban users or delete messages.
    soft_mode: bool = True

    # These are the users who will get tagged if spam is detected.
    # Example: ["username234"]
    admin_usernames: List[str] = field(default_factory=list)

    # ID of the chat/channel where spam messages will be forwarded.
    # Easiest way to get this ID: https://gist.github.com/mraaroncruz/e76d19f7d61d59419002db54030ebe35
    # May be omitted if you don't want to see which messages are considered as spam.
    # You can still see them in "Recent actions" of the chat if you're an admin.
    spam_dump: int | None = None

    # How long the ban should last.
    # Recommendation: don't ban users for a long time.
    #   Wrongful bans might happen, and most spammers don't come back anyway.
    ban_duration: timedelta = timedelta(seconds=30)

    # List of lowercase words/phrases that will make the bot skip the spam check.
    good_words: List[str] | None = None

    # List of lowercase words/phrases that will flag the message as spam instantly.
    bad_words: List[str] | None = None


# Add your chat here to allow the bot to work there.
allowed_chats = {
    "feature_sliced": ChatSettings(
        spam_dump=-1002296187217,
        ban_duration=timedelta(seconds=30),
        soft_mode=False,
        admin_usernames=["illright"],
        good_words=[
            "fsd",
            "фсд",
            "widget",
            "виджет",
            "entity",
            "entities",
            "энтити",
            "feature",
            "фич",
            "@MmrizFastTrackBot",
        ],
        bad_words=["сидеть без денег", "легких денег"],
    ),
    "goedemorgen_walks": ChatSettings(
        soft_mode=False,
    ),
    # "your_chat_link_without_@": ChatSettings(),
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("antispam-telegram-bot-token")],
    enable_memory_snapshot=True,
    scaledown_window=2,
)
class Model:
    model_path = os.path.join(remote_project_path, classifier_path)

    @modal.enter(snap=True)
    def load_model_weights(self):
        """Load the model into memory. `snap=True` creates a memory snapshot to speed up further runs."""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=1
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    @modal.enter(snap=False)
    def setup_bot(self):
        """Register event handlers for the bot."""
        self.dp = Dispatcher()

        @self.dp.chat_member(ChatMemberUpdatedFilter(IS_NOT_MEMBER >> IS_MEMBER))
        def on_new_user(event: types.chat_member_updated.ChatMemberUpdated):
            """When someone joins, they are considered unchecked until they write a message."""
            logger.info(f"New unchecked user: {event.new_chat_member.user.id}")
            unchecked_users[event.new_chat_member.user.id] = event.date

        @self.dp.my_chat_member(ChatMemberUpdatedFilter(IS_NOT_MEMBER >> IS_MEMBER))
        async def on_me_added_to_chat(
            event: types.chat_member_updated.ChatMemberUpdated,
        ):
            """Leave chats where the bot is not expected to run."""
            if (
                event.chat.id not in allowed_chats
                and event.chat.username not in allowed_chats
            ):
                logger.info(f"Added to unknown chat {event.chat.id}, leaving")
                await self.bot.leave_chat(event.chat.id)

        @self.dp.message()
        async def on_message(message: types.Message):
            """Check the first message of every user and ban them if it's detected as spam."""
            if message.from_user.id not in unchecked_users:
                logger.info(f"User {message.from_user.id} is checked, skipping")
                return

            chat_settings = allowed_chats.get(message.chat.id) or allowed_chats.get(
                message.chat.username
            )
            if chat_settings is None:
                logger.error(f"Message in unknown chat, ID: {message.chat.id}")
                return

            try:
                check_passed = False
                message_text = self.get_text_from_photo(message) or message.text

                has_bad_words = message_text is not None and self.contains_words(
                    message_text, chat_settings.bad_words
                )
                has_good_words = message_text is not None and self.contains_words(
                    message_text, chat_settings.good_words
                )
                has_telegram_links = message_text is not None and (
                    "t.me/" in message_text
                    or any(
                        entity.type == "text_link" and "t.me/" in entity.url
                        for entity in (message.entities or [])
                    )
                )
                has_buttons = message.reply_markup is not None
                is_forwarded_from_channel = (
                    message.forward_from_chat is not None
                    and message.forward_from_chat.type == "channel"
                )
                is_story = message.story is not None

                if has_good_words:
                    check_passed = True
                    logger.info(
                        f"Good words found in message from {message.from_user.id}"
                    )
                elif has_bad_words:
                    check_passed = False
                    logger.info(
                        f"Bad words found in message from {message.from_user.id}"
                    )
                elif has_telegram_links:
                    check_passed = False
                    logger.info(
                        f"Telegram links found in message from {message.from_user.id}"
                    )
                elif has_buttons:
                    check_passed = False
                    logger.info(f"Buttons found in message from {message.from_user.id}")
                elif is_forwarded_from_channel:
                    check_passed = False
                    logger.info(
                        f"Message forwarded from a channel by {message.from_user.id}"
                    )
                elif is_story:
                    check_passed = False
                    logger.info(f"Story sent by user {message.from_user.id}")
                elif self.is_spam.local(message_text):
                    check_passed = False
                    logger.info(f"Spam detected from user {message.from_user.id}")
                elif message_text is None:
                    logger.info("Message has no text, skipping")
                    return
                elif len(message_text) <= 2 or re.match(r"[^\w]+$", message_text):
                    logger.info("Message is too short to check, skipping")
                    return
                else:
                    check_passed = True
                    logger.info(f"Message from {message.from_user.id} is not spam")

                if check_passed:
                    logger.info(f"User {message.from_user.id} is now checked")
                    unchecked_users.pop(message.from_user.id)
                else:
                    if chat_settings.spam_dump is not None:
                        await message.forward(
                            chat_id=chat_settings.spam_dump, disable_notification=True
                        )

                    if chat_settings.soft_mode:
                        await message.reply(
                            f"Spam! {' '.join(f'@{username}' for username in chat_settings.admin_usernames)}"
                        )
                    else:
                        await self.bot.ban_chat_member(
                            chat_id=message.chat.id,
                            user_id=message.from_user.id,
                            until_date=chat_settings.ban_duration,
                        )
                        await message.delete()
            except Exception as e:
                logger.error(f"Error while processing message: {e}")

        if "TELEGRAM_BOT_TOKEN" in os.environ:
            self.bot = Bot(
                token=os.environ["TELEGRAM_BOT_TOKEN"],
                default=DefaultBotProperties(parse_mode=ParseMode.HTML),
            )

    @modal.fastapi_endpoint(method="POST")
    async def process_update(
        self,
        update: dict,
        x_telegram_bot_api_secret_token: Annotated[str | None, Header()] = None,
    ) -> None | dict:
        """Accept an update from Telegram's API and verify that it's legit."""

        if not hmac.compare_digest(
            x_telegram_bot_api_secret_token, os.environ["EXTRA_SECURITY_TOKEN"]
        ):
            logger.error("Wrong secret token!")
            return {"status": "error", "message": "Wrong secret token!"}

        logger.info(f"Received update: {update}")
        telegram_update = types.Update(**update)
        try:
            await self.dp.feed_webhook_update(bot=self.bot, update=telegram_update)
        except Exception as e:
            logger.error(e, exc_info=True)

    @modal.method()
    def is_spam(self, message: str):
        message = clean_text(message)
        encoding = self.tokenizer(
            message,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask).logits
            pred = torch.sigmoid(outputs).cpu().numpy()[0][0]

        logger.debug(f"Predicted spam score: {pred}")
        return int(pred >= 0.5)

    @staticmethod
    def contains_words(text: str, words: List[str] | None):
        lowercase_text = text.lower()
        return any(word in lowercase_text for word in (words or []))

    async def get_text_from_photo(self, message: types.Message) -> str | None:
        """Extract text from photo using OCR and add the caption."""
        if message.photo is None:
            return None

        highest_res_photo = max(message.photo, key=lambda p: p.width * p.height)
        file_path = (await self.bot.get_file(highest_res_photo.file_id)).file_path
        if file_path is None:
            return message.caption

        photo_bytes = io.BytesIO()
        await self.bot.download_file(file_path, destination=photo_bytes)
        photo_bytes.seek(0)

        text_on_photo = pytesseract.image_to_string(Image.open(photo_bytes), lang="rus")
        if text_on_photo:
            return "\n".join((text_on_photo, message.caption or ""))
        else:
            return message.caption


# Run with `uv run modal run main` to check a single message.
@app.local_entrypoint()
def check_single_message():
    message = ""

    print("Spam!" if Model().is_spam.local(message) else "Not spam")
    print(
        "Has bad words"
        if Model.contains_words(message, allowed_chats["feature_sliced"].bad_words)
        else "No bad words"
    )
    print(
        "Has good words"
        if Model.contains_words(message, allowed_chats["feature_sliced"].good_words)
        else "No good words"
    )
