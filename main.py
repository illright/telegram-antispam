import re
import hmac
import logging
import os
from datetime import timedelta, datetime
from typing import Annotated, List
from dataclasses import dataclass

import modal
from fastapi import Header
from aiogram import types

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1")
    .pip_install("transformers==4.44.2")
    .pip_install("fastapi==0.114.2")
    .pip_install("pydantic==2.8.2")
    .pip_install("aiogram==3.13.0")
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
class ChatSettings:
    soft_mode: bool
    admin_usernames: List[str]
    spam_dump: int | None = None
    ban_duration: timedelta = timedelta(days=7)
    good_words: List[str] | None = None
    bad_words: List[str] | None = None


@dataclass
class UncheckedUserDossier:
    join_time: datetime
    first_reaction_time: datetime | None = None


allowed_chats = {
    "feature_sliced": ChatSettings(
        spam_dump=-1002296187217,
        ban_duration=timedelta(hours=2),
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
        bad_words=["заработка", "дoхoд"],
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
    pt_save_directory = "./ruSpamNS_V1"
    model_name = "NeuroSpaceX/ruSpamNS_v1"

    @modal.build()
    def download_model_to_folder(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_token = os.environ["HF_TOKEN"]

        model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=1, token=hf_token
            )
            .to(device)
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        model.save_pretrained(self.pt_save_directory)
        tokenizer.save_pretrained(self.pt_save_directory)

    @modal.enter(snap=True)
    def load_model_weights(self):
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.pt_save_directory, num_labels=1
            )
            .to("cpu")
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.pt_save_directory)

    @modal.enter(snap=False)
    def setup_bot(self):
        self.dp = Dispatcher()

        @self.dp.chat_member(ChatMemberUpdatedFilter(IS_NOT_MEMBER >> IS_MEMBER))
        def on_new_user(event: types.chat_member_updated.ChatMemberUpdated):
            unchecked_users[event.new_chat_member.user.id] = UncheckedUserDossier(
                join_time=event.date
            )

        @self.dp.my_chat_member(ChatMemberUpdatedFilter(IS_NOT_MEMBER >> IS_MEMBER))
        async def on_me_added_to_chat(
            event: types.chat_member_updated.ChatMemberUpdated,
        ):
            if (
                event.chat.id not in allowed_chats
                and event.chat.username not in allowed_chats
            ):
                logger.info(f"Added to unknown chat {event.chat.id}, leaving")
                await self.bot.leave_chat(event.chat.id)

        @self.dp.message_reaction()
        async def on_reaction(event: types.MessageReactionUpdated):
            if event.user.id not in unchecked_users:
                return

            chat_settings = allowed_chats.get(event.chat.id) or allowed_chats.get(
                event.chat.username
            )
            if chat_settings is None:
                logger.error(f"Reaction in unknown chat, ID: {event.chat.id}")
                return

            user_dossier = unchecked_users[event.user.id]
            if user_dossier.first_reaction_time is None:
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
            if message.from_user.id not in unchecked_users or message.text is None:
                return

            chat_settings = allowed_chats.get(message.chat.id) or allowed_chats.get(
                message.chat.username
            )
            if chat_settings is None:
                logger.error(f"Message in unknown chat, ID: {message.chat.id}")
                return

            has_bad_words = self.contains_words(message.text, chat_settings.bad_words)
            has_good_words = self.contains_words(message.text, chat_settings.good_words)
            if not has_bad_words and (
                has_good_words or not self.is_spam.local(message.text)
            ):
                unchecked_users.pop(message.from_user.id)
            else:
                logger.info(f"Spam detected from user {message.from_user.id}")
                try:
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
        message = self.clean_text(message)
        encoding = self.tokenizer(
            message,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to("cpu")
        attention_mask = encoding["attention_mask"].to("cpu")

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask).logits
            pred = torch.sigmoid(outputs).cpu().numpy()[0][0]

        logger.debug(f"Predicted spam score: {pred}")
        return int(pred >= 0.5)

    @staticmethod
    def clean_text(text: str):
        text = re.sub(r"\w+", lambda m: Model.recover_cyrillic_in_word(m.group()), text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^А-Яа-я0-9 ]+", " ", text)
        text = text.lower().strip()
        return text

    @staticmethod
    def recover_cyrillic_in_word(word: str):
        """Convert common Latin substitutions of Cyrillic letters back to the original ones, for example, "с" -> "c"."""
        translation_table = str.maketrans(
            "cyeaopxurkbnmETYOPAHKXCBM",
            "суеаорхигкьпмЕТУОРАНКХСВМ",
        )
        cyrillic_letter_pattern = re.compile(r"[а-яё]", re.IGNORECASE)
        if cyrillic_letter_pattern.search(word):
            return word.translate(translation_table)
        else:
            return word

    @staticmethod
    def contains_words(text: str, words: List[str] | None):
        lowercase_text = text.lower()
        return words is None or any(word in lowercase_text for word in words)


# uv run modal run main
@app.local_entrypoint()
def check_single_message():
    message = ""

    print("Spam!" if Model().is_spam.local(message) else "Not spam")
