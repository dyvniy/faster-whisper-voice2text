import torch
import pyaudio
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datetime import datetime
from time import ctime

# Загрузка модели и процессора для русского языка
MODEL_ID = ["jonatasgrosman/wav2vec2-large-xlsr-53-russian",
            "bond005/wav2vec2-large-ru-golos",                  # good
            "facebook/wav2vec2-large-960h-lv60-self",             # no
            "RuiqianLi/wav2vec2-large-960h-lv60-self-4-gram_fine-tune_real_29_Jun" # eng
            ]
MODEL_NUMBER = 1

print("Загрузка процессора...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID[MODEL_NUMBER])
print("Загрузка модели...")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID[MODEL_NUMBER])
print("Модель успешно загружена!")

# Параметры записи аудио
FORMAT = pyaudio.paInt16  # Формат аудио
CHANNELS = 1              # Количество каналов (моно)
RATE = 16000              # Частота дискретизации (должна совпадать с моделью)
CHUNK = 16000             # Размер блока данных (1 секунда)
WINDOW_SIZE = 2           # Размер окна для распознавания (в секундах)
SILENCE_THRESHOLD = 50    # Порог громкости для игнорирования тишины

# Открываем файл для записи текста
output_file = "transcription.txt"
with open(output_file, "a", encoding="utf-8") as f:
    f.write("Начало распознавания:\n")

def transcribe_audio(audio_data):
    """Передает аудиоданные в модель wav2vec для распознавания."""
    # Преобразуем аудио в формат, подходящий для модели
    input_values = processor(audio_data, sampling_rate=RATE, return_tensors="pt").input_values

    # Явно преобразуем данные в float32
    input_values = input_values.to(torch.float32)

    # Распознаем речь
    with torch.no_grad():
        logits = model(input_values).logits

    # Декодируем предсказания
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription.strip()  # Убираем лишние пробелы

def correct_text(text):
    """Простая функция для коррекции текста."""
    # Пример: исправление часто встречающихся ошибок
    corrections = {
        "превет": "привет",
        "делаа": "дела",
        "какд дела": "как дела",
        "спасиб": "спасибо",
        "покаа": "пока"
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

def main():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Начало записи...")

    buffer = []  # Буфер для хранения аудиоданных
    last_transcription = ""  # Последний распознанный текст

    try:
        while True:
            # Чтение данных с микрофона
            data = stream.read(CHUNK)
            audio_chunk = np.frombuffer(data, dtype=np.int16)

            # Вычисляем средний уровень громкости
            volume = np.abs(audio_chunk).mean()

            # Игнорируем тихие фрагменты
            if volume < SILENCE_THRESHOLD:
                continue

            # Добавляем данные в буфер
            buffer.append(audio_chunk)

            # Если буфер содержит достаточно данных для обработки
            if len(buffer) >= WINDOW_SIZE:
                # Объединяем данные из буфера
                audio_data = np.concatenate(buffer[:WINDOW_SIZE])
                buffer = buffer[WINDOW_SIZE:]  # Очищаем использованные данные

                # Распознаем текст
                transcription = transcribe_audio(audio_data)

                # Корректируем текст
                transcription = correct_text(transcription)

                # Если текст изменился, выводим его
                if transcription != last_transcription and transcription:
                    print(f"{transcription}")  # Выводим текст на новой строке
                    last_transcription = transcription

                    # Записываем текст в файл
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(ctime() + f" {transcription}\n")

    except KeyboardInterrupt:
        print("\nЗавершение работы...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()