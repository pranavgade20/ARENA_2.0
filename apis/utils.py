import csv
import logging
import os
import re
import tempfile
import threading
import time
import traceback
from datetime import datetime
from urllib.parse import urlparse

import PyPDF2
import ffmpeg
import html2text as html2text
import ocrmypdf
import openai
import pandas as pd
import requests
import whisper
import yt_dlp.YoutubeDL
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, filters, MessageHandler, CallbackQueryHandler
import subprocess as sub

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/p/Downloads/pranavg-dfb6e144f64b.json'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
temp_path = tempfile.gettempdir()
openai.api_key = "sk-"


print('Initializing whisper model...')
model = whisper.load_model("base")


def summarize(text) -> str:
    lock.acquire()
    if len(text) > 3500:  # hacky way to summarize a large piece of text
        return summarize(summarize(text[:3200]) + '\n' + text[2900:])
    response = openai.Completion.create(engine="text-davinci-003", prompt=f'Summarize this:\n\n{text}\n\ntl;dr:',
                                        temperature=0.3,
                                        max_tokens=512,
                                        top_p=1,
                                        frequency_penalty=0.1,
                                        presence_penalty=0.1,
                                        stop=["\n"])
    return response["choices"][0]["text"]


def get_title(text) -> str:
    lock.acquire()
    if len(text) > 3500:  # hacky way to summarize a large piece of text
        text = summarize_parallel(text)
    response = openai.Completion.create(engine="text-davinci-003", prompt=f'Write a title for:\n\n{text}\n\nTitle:',
                                        temperature=0.3,
                                        max_tokens=128,
                                        top_p=1,
                                        frequency_penalty=0.1,
                                        presence_penalty=0.1,
                                        stop=["\n"])
    return response["choices"][0]["text"]


lock = threading.Lock()


def lock_releaser():
    while True:
        try:
            time.sleep(1)
            lock.release()
        except RuntimeError:
            pass


threading.Thread(target=lock_releaser, daemon=True).start()


def summarize_parallel(text) -> str:
    def summarize_and_write(t, arr, i):
        arr[i] = summarize(t)

    if len(text) > 3500:  # hacky way to summarize a large piece of text
        broken_text = [text[i:i + 3000] for i in range(0, len(text), 3000)]
        rets = ['' for i in range(len(broken_text))]
        threads = [threading.Thread(target=summarize_and_write, args=(broken_text[i], rets, i)) for i in
                   range(len(broken_text))]
        any(t.start() for t in threads)
        any(t.join() for t in threads)
        print(rets)
        return summarize_parallel(''.join(rets))
    else:
        return summarize(text)


def detect_labels(path) -> [str]:
    """Given a file path of an image, label it using Google Cloud Vision API"""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return [label.description for label in labels]


def detect_document(path):
    """Given an image path, extract text from it using Google Cloud Vision API"""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return '\n\n\n'.join(['\n\n'.join(['\n'.join(
        [' '.join([''.join([s.text for s in word.symbols]) for word in paragraph.words]) for paragraph in
         block.paragraphs]) for block in page.blocks]) for page in response.full_text_annotation.pages])


def handle_pdf_file(pdf_path):
    """Given a pdf, use ocrmypdf to extract text from it, ans summarize the result"""
    ocrmypdf.ocr(pdf_path, pdf_path + '.1', skip_text=True)
    reader = PyPDF2.PdfReader(pdf_path + '.1')
    text = "\n\n".join([page.extract_text() for page in reader.pages])
    return summarize_parallel(text)


async def get_updates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logs = read_slack_logs()
    if len(logs) == 0:
        await send_text("no messages!", update)
        return
    for team in logs:
        log = f'[{team}]\n\n'
        for channel in logs[team]:
            summary = summarize_parallel(logs[team][channel])
            log += f'[{channel}]:\n{summary}\n\n'
        await send_text(log, update)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! I can summarize content for you - send me anything, and I'll try to give you a summary, or try /help for detailed information.", )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        '''Hello! I can summarize audio, videos, images of documents, text, PDFs, webpages and youtube videos for you.

        You can also use /echo and I'll repeat what you say.
        '''
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    def is_uri(x):
        try:
            result = urlparse(x)
            return all([result.scheme, result.netloc])
        except:
            return False

    if is_uri(update.message.text):
        url = urlparse(update.message.text)
        if str(url.hostname) in ['www.youtube.com', 'www.youtu.be', 'youtube.com', 'youtu.be']:
            file_path = temp_path + '/' + str(url.path.split('/')[-1]) + '.wav'
            yt_dlp._real_main(
                ['--no-playlist', '--ignore-errors', '--output', file_path, '--extract-audio', '--audio-format', 'wav',
                 update.message.text])
            return await handle_audio_file(file_path, update)
        resp = requests.get(update.message.text, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:104.0) Gecko/20100101 Firefox/104.0"},
                            allow_redirects=True)
        if 'Content-Type' in resp.headers:
            if resp.headers['Content-Type'].startswith('text/html'):
                print("started summarizing")
                await send_text(summarize_parallel(html2text.html2text(str(resp.content))), update)
                return
            elif resp.headers['Content-Type'].startswith('application/pdf') or resp.headers['Content-Type'].startswith(
                    'video/') or resp.headers['Content-Type'].startswith('image/') or resp.headers[
                'Content-Type'].startswith('audio/'):
                file_path = temp_path + "/" + resp.url.split('/')[-1]
                with open(file_path, 'wb') as f:
                    f.write(resp.content)
                await handle_file_path(file_path, update)
                return
        else:
            await send_text("content type not found or not supported", update)
            return
    else:
        await send_text(summarize_parallel(update.message.text), update)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo text to user"""
    await send_text(update.message.text, update)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()

    if query.data == 'Summarize':
        summary = summarize_parallel(query.message.text)
        await send_text(summary, query)
    # elif query.data == 'Send to trilium':
    #     title = get_title(query.message.text).strip() or 'untitled'
    #     if title[0] == '"' and title[-1] == '"':
    #         title = title[1:-1]
    #     trilium.create_note(trilium.get_calendar_days(datetime.now().strftime("%Y-%m-%d"))['noteId'], title, "text",
    #                         content=query.message.text)
    #     await query.edit_message_reply_markup(InlineKeyboardMarkup(
    #         [[InlineKeyboardButton(action, callback_data=action)] for action in ['Summarize', f"Sent to '{title}'"]]))
    else:
        await send_text("Function not implemented yet :/", query)


async def handle_attachment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file_path = download_attachment_to_file(update)

    await handle_file_path(file_path, update)


async def handle_file_path(file_path, update):
    p = sub.Popen(['file', '-S', '--mime-type', f'{file_path}'], stdout=sub.PIPE, stderr=sub.PIPE)
    file_type, errors = p.communicate()
    file_type = file_type.split(b' ')[1]
    if b'application/pdf' in file_type:
        summary = handle_pdf_file(file_path)
        await send_text(summary, update)
    elif b'video/' in file_type:
        file_path = extract_audio_from_video(file_path)
        await handle_audio_file(file_path, update)
        return
    elif b'audio/' in file_type:
        await handle_audio_file(file_path, update)
        return
    elif b'image/' in file_type:
        await handle_image_file(file_path, update)
        return
    else:
        await send_text("Function not implemented yet :/", update)


async def handle_voice_attachment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file_path = download_attachment_to_file(update)

    await handle_audio_file(file_path, update)


async def handle_image_file(file_path, update):
    labels = detect_labels(file_path)
    if 'Font' in labels:
        await send_text(detect_document(file_path), update)
        return
    result = ', '.join(labels)
    await send_text(result, update)


async def handle_audio_file(file_path, update):
    result = whisper.transcribe(model, file_path, fp16=False)
    print(result)
    await send_text(result['text'], update)


def download_attachment_to_file(update):
    file_id = update.message.effective_attachment[-1].file_id if isinstance(update.message.effective_attachment,
                                                                            list) else update.message.effective_attachment.file_id
    resp = requests.get(
        f"https://api.telegram.org/bottoken_goes_here/getFile?file_id={file_id}").json()[
        "result"]
    r = requests.get(
        f'https://api.telegram.org/file/bottoken_goes_here/{resp["file_path"]}',
        allow_redirects=True)
    file_path = temp_path + "/" + resp['file_path'].split('/')[-1]
    with open(file_path, 'wb') as f:
        f.write(r.content)
    return file_path


async def send_text(result, update):
    text_reply = result if result else "EMPTY MESSAGE"
    reply_markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton(action, callback_data=action)] for action in ['Summarize', 'Send to trilium']])
    await update.get_bot().send_message(
        chat_id=update.message.chat_id,
        text=text_reply,
        reply_markup=reply_markup,
        reply_to_message_id=update.message.message_id
    )


async def handle_video_attachment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file_path = download_attachment_to_file(update)
    file_path = extract_audio_from_video(file_path)

    await handle_audio_file(file_path, update)


def extract_audio_from_video(file_path):
    out, _ = (
        ffmpeg.input(file_path, threads=0)
        .output(f"{file_path}.wav", acodec="pcm_s16le", ac=1)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    file_path = f"{file_path}.wav"
    return file_path


if __name__ == '__main__':
    threading.Thread(target=start_slack_app, daemon=True).start()
    application = ApplicationBuilder().token('token_goes_here').build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("echo", echo))
    application.add_handler(CommandHandler("update", get_updates))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler((filters.VOICE | filters.AUDIO) & ~filters.COMMAND, handle_voice_attachment))
    application.add_handler(
        MessageHandler((filters.VIDEO | filters.VIDEO_NOTE) & ~filters.COMMAND, handle_video_attachment))
    application.add_handler(MessageHandler(filters.ATTACHMENT & ~filters.COMMAND, handle_attachment))

    application.run_polling()
