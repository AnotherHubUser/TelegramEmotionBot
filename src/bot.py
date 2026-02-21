from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from pathlib import Path
from emotion_classifier import EmotionClassifier

from config import TOKEN
from logger import get_logger, with_log_context

logger = get_logger(__name__)


class Bot:
    def __init__(self, token=TOKEN):
        self.classifier = EmotionClassifier()

        self.application = Application.builder().token(token).build()
        
        logger.debug("add command handlers")
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("emote", self.emote_command))
        
        logger.debug("add callback handlers")
        self.application.add_handler(CallbackQueryHandler(self.button_callback))

        logger.debug("add message handlers")
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        self.application.add_handler(MessageHandler(filters.ALL, self.handle_unknown))
        
        logger.debug("add error handlers")
        self.application.add_error_handler(self.error_handler)

    def run(self):
        logger.info("run, bot, run")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)     

    @with_log_context
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        logger.info(f"started by {user.id}({user.full_name})")
        await update.message.reply_text(
            f"Darova, {user.id}({user.first_name})! 👋\n"
        )
        logger.debug(f"greeted with user {user.id}({user.full_name})")

    @with_log_context
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"help requested by {update.effective_user.id}({update.effective_user.full_name}).")
        await update.message.reply_text(
            "Send voice message and I will classify its emotions\n",
            parse_mode="Markdown"
        )
        logger.debug(f"helped replied to {update.effective_user.id}({update.effective_user.full_name}).")

    @with_log_context
    async def emote_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        logger.info(f"{user.id}({user.full_name}) started emotion transformation conversation")

        keyboard = [
            [
                InlineKeyboardButton("😊 positive", callback_data="positive"),
                InlineKeyboardButton("😠 angry", callback_data="angry"),
            ],
            [
                InlineKeyboardButton("😢 sad", callback_data="sad"),
                InlineKeyboardButton("😐 neutral", callback_data="neutral"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        context.user_data['state'] = 'waiting_emotion'
        
        await update.message.reply_text(
            "Choose emotion to transfrom your next voice message:",
            reply_markup=reply_markup
        )

    @with_log_context
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        user = update.effective_user
        callback_data = query.data

        logger.info(f"user {user.id}({user.full_name}) pressed {callback_data} button")

        if context.user_data.get('state') == 'waiting_emotion':
            context.user_data['target_emotion'] = callback_data
            context.user_data['state'] = 'waiting_voice'
            
            await query.edit_message_text(f"Now send voice message!")
            logger.info(f"User {user.id}({user.full_name}) selected emotion: {callback_data}")

        else:
            await query.edit_message_text("use /process firstly")
            logger.info(f"User {user.id}({user.full_name}) has broken pipeline somehow")

    @with_log_context
    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        logger.info(f"get voice message from {user.id}({user.full_name})")
        
        await update.message.reply_text("🎤 Получил голосовое! Начинаю обработку...")
        
        try:
            # model proccessing staff here
            file_size = update.message.voice.file_size
            duration = update.message.voice.duration       
            logger.debug(f"Size of your message {file_size}, duration {duration}")

            file_path = await self.download_audio(update, user.id)
 
            logger.debug(f"voice file succesfully loaded")

            state = context.user_data.get('state')
            target_emotion = context.user_data.get('target_emotion')
            if state == 'waiting_voice' and target_emotion:
                await update.message.reply_text("trying to convert your voice into {target_emotion}")

                # voice converting here

                context.user_data.pop('state', None)
                context.user_data.pop('target_emotion', None)

                logger.debug(f"before returning file_path")
                await update.message.reply_voice(file_path)
                logger.debug(f"after returning filepath")
                # await update.message.reply_text()

            else:
                emotions2probs = [(emotion, prob) for emotion, prob in self.classifier.predict_with_scores(file_path).items()]
                emotions2probs.sort(key=lambda x: -x[1])

                message = ""
                for emotion, prob in emotions2probs:
                    message += f"{emotion}:\t\t{prob * 100:.2f}%\n"

                await update.message.reply_text(message)
                logger.info(f"replied to user {user.id}({user.full_name}) with his voice score")
                        
        except Exception as e:
            logger.error(f"Exception while proccesing voice message from {user.id}({user.name})\n{e}")
            await update.message.reply_text(
                f"enternal error. HZ v chem delo. {e}"
            )

    async def download_audio(self, update: Update, user_id: int):
        voice_file = await update.message.effective_attachment.get_file()
        file_id = update.message.voice.file_id

        directory_path = Path(f"data/heap/{user_id}")
        directory_path.mkdir(parents=True, exist_ok=True)

        file_path = directory_path / Path(f"{file_id}.ogg")
        await voice_file.download_to_drive(file_path)
        return file_path


    @with_log_context
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"user {update.effective_user.id} send non voice message")
        logger.info(f"update_id is {update.update_id}")
        await update.message.reply_text(
            f"Izvini, ya ne umeu raspoznavat' ne golosovie soobscheniya"
        )
        logger.debug(f"replied to {update.effective_user.id} to non voice message")

    @with_log_context
    async def handle_unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"user {update.effective_user.id} send non voice message")
        await update.message.reply_text(
            f"Izvini, ya ne umeu raspoznavat' ne golosovie soobscheniya"
        )
        logger.debug(f"replied to {update.effective_user.id} to non voice message")

    @with_log_context
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Error: {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                f"enternal error. HZ v chem delo. {context.error}"
            )


def main():
    pass
 
if __name__ == "__main__":
    main()