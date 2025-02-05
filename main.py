import hmac
import logging
import os
from datetime import timedelta, datetime
from typing import Annotated, List
from dataclasses import dataclass, field

import modal
from fastapi import Header
from aiogram import types

from clean_text.v7_tiny import clean_text

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1")
    .pip_install("transformers==4.47.1")
    .pip_install("fastapi==0.115.6")
    .pip_install("pydantic==2.8.2")
    .pip_install("aiogram==3.16.0")
)

with image.imports():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from aiogram import Bot, Dispatcher, types
    from aiogram.enums import ParseMode
    from aiogram.client.default import DefaultBotProperties
    from aiogram.filters import IS_MEMBER, IS_NOT_MEMBER, ChatMemberUpdatedFilter


app = modal.App(name="antispam", image=image)
unchecked_users = modal.Dict.from_name("unchecked_users", create_if_missing=True)


@dataclass
class UncheckedUserDossier:
    """Before a user proves that they're not a spammer, this data is stored on them."""

    join_time: datetime
    first_reaction_time: datetime | None = None


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
        ],
        bad_words=["сидеть без денег", "легких денег"],
    )
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.cls(
    image=image,
    secrets=[
        modal.Secret.from_name("antispam-hf-token"),
        modal.Secret.from_name("antispam-telegram-bot-token"),
    ],
    enable_memory_snapshot=True,
    container_idle_timeout=2,
)
class Model:
    pt_save_directory = "./ruSpamNS_v7_tiny"
    model_name = "NeuroSpaceX/ruSpamNS_v7_tiny"

    @modal.build()
    def download_model_to_folder(self):
        """Download the model to the container. Runs once when the app is deployed."""
        hf_token = os.environ["HF_TOKEN"]

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1, token=hf_token
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        model.save_pretrained(self.pt_save_directory)
        tokenizer.save_pretrained(self.pt_save_directory)

    @modal.enter(snap=True)
    def load_model_weights(self):
        """Load the model into memory. `snap=True` creates a memory snapshot to speed up further runs."""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pt_save_directory, num_labels=1
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.pt_save_directory)

    @modal.enter(snap=False)
    def setup_bot(self):
        """Register event handlers for the bot."""
        self.dp = Dispatcher()

        @self.dp.chat_member(ChatMemberUpdatedFilter(IS_NOT_MEMBER >> IS_MEMBER))
        def on_new_user(event: types.chat_member_updated.ChatMemberUpdated):
            """When someone joins, they are considered unchecked until they write a message."""
            logger.info(f"New unchecked user: {event.new_chat_member.user.id}")
            unchecked_users[event.new_chat_member.user.id] = UncheckedUserDossier(
                join_time=event.date
            )

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

        @self.dp.message_reaction()
        async def on_reaction(event: types.MessageReactionUpdated):
            """Ban users if they don't write messages and leave more than 2 reactions in 5 minutes."""
            if len(event.old_reaction) > len(event.new_reaction):
                logger.info("Reaction removed, skipping")
                return

            if event.user.id not in unchecked_users:
                logger.info(f"User {event.user.id} is checked, skipping")
                return

            chat_settings = allowed_chats.get(event.chat.id) or allowed_chats.get(
                event.chat.username
            )
            if chat_settings is None:
                logger.error(f"Reaction in unknown chat, ID: {event.chat.id}")
                return

            user_dossier = unchecked_users[event.user.id]
            if user_dossier.first_reaction_time is None:
                logger.info(f"First reaction from {event.user.id}")
                user_dossier.first_reaction_time = event.date
                unchecked_users[event.user.id] = user_dossier
            else:
                time_between_reactions = event.date - user_dossier.first_reaction_time
                if time_between_reactions < timedelta(minutes=5):
                    logger.info(f"Spam reactions from {event.user.username}")
                    await self.bot.ban_chat_member(
                        chat_id=event.chat.id,
                        user_id=event.user.id,
                        until_date=chat_settings.ban_duration,
                    )

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

                message_text = message.caption or message.text
                if message_text is None:
                    logger.info("Message has no text, skipping")
                    return

                has_bad_words = self.contains_words(
                    message_text, chat_settings.bad_words
                )
                has_good_words = self.contains_words(
                    message_text, chat_settings.good_words
                )
                has_telegram_links = "t.me/" in message_text or any(
                    entity.type == "text_link" and "t.me/" in entity.url
                    for entity in (message.entities or [])
                )
                has_buttons = message.reply_markup is not None
                is_forwarded_from_channel = (
                    message.forward_from_chat is not None
                    and message.forward_from_chat.type == "channel"
                )

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
                elif self.is_spam.local(message_text):
                    check_passed = False
                    logger.info(f"Spam detected from user {message.from_user.id}")
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

    @modal.web_endpoint(method="POST")
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
