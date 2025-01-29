#If you have any questions refer to my via telegram @biodata

import os
import re
import cv2
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
import img2pdf

import openai  # pip install --upgrade openai

# Замените на свой реальный ключ
openai.api_key = "YOUR_OPEN_AI_API_KEY"


def ask_chatgpt_for_fio(debug_text):
    system_message = (
        "Ты - помощник, который умеет возвращать JSON. "
        "Тебе дается текст, разбитый построчно. "
        "Нужно определить, есть ли в этих строках упоминания ФИО "
        "и вернуть список объектов вида {\"line\": <int>, \"words\": [\"...\"]}.\n\n"
        "Если нет упоминаний ФИО, верни пустой список []."
    )

    user_message = (
        "Ниже текст, где каждая строка имеет номер.\n"
        "Нужно найти, где упоминаются ФИО (пациента, врача и т.д.), "
        "и вернуть массив JSON со строкой и словами, которые относятся к ФИО.\n\n"
        f"{debug_text}\n\n"
        "Ответь ТОЛЬКО структурой JSON без пояснений."
    )

    # ВАЖНО: Используем ChatCompletion, а не chat_completions
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    # Забираем контент
    return response["choices"][0]["message"]["content"]


def parse_chatgpt_json(answer):
    import json
    try:
        data = json.loads(answer)
        if isinstance(data, list):
            return data
        else:
            return []
    except:
        return []


def collect_lines_and_debug_text(ocr_data):
    lines_map = {}
    n = len(ocr_data["text"])
    for i in range(n):
        w = ocr_data["text"][i].strip()
        if not w:
            continue
        block = ocr_data["block_num"][i]
        line = ocr_data["line_num"][i]
        lines_map.setdefault((block, line), []).append(i)

    line_keys = sorted(lines_map.keys())
    debug_lines = []
    for idx, key in enumerate(line_keys, start=1):
        words_indices = lines_map[key]
        parts = []
        for w_i in words_indices:
            parts.append(ocr_data["text"][w_i].strip())
        line_str = " ".join(parts)
        debug_lines.append(f"{idx}) {line_str}")

    debug_text = "\n".join(debug_lines)
    return lines_map, debug_text, line_keys


def redact_image_line_by_line_with_chatgpt(img):
    config = r'--oem 3 --psm 6'
    ocr_data = pytesseract.image_to_data(
        img, output_type=Output.DICT,
        config=config, lang='rus'
    )

    lines_map, debug_text, line_keys = collect_lines_and_debug_text(ocr_data)

    gpt_answer = ask_chatgpt_for_fio(debug_text)
    parsed_data = parse_chatgpt_json(gpt_answer)

    for info in parsed_data:
        line_num = info.get("line")
        fio_words = info.get("words", [])
        if not isinstance(line_num, int):
            continue
        line_idx = line_num - 1
        if line_idx < 0 or line_idx >= len(line_keys):
            continue

        block_line = line_keys[line_idx]
        word_indices = lines_map[block_line]

        for fiow in fio_words:
            fiow_lower = fiow.strip().lower()
            if not fiow_lower:
                continue
            for w_i in word_indices:
                text_w = ocr_data["text"][w_i].strip().lower()
                if text_w == fiow_lower:
                    x = ocr_data["left"][w_i]
                    y = ocr_data["top"][w_i]
                    w = ocr_data["width"][w_i]
                    h = ocr_data["height"][w_i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

    return img


def process_png(filename):
    img = cv2.imread(filename)
    if img is None:
        print(f"Не удалось открыть PNG: {filename}")
        return

    redacted = redact_image_line_by_line_with_chatgpt(img)
    out_name = filename.replace('.png', '_result.png')
    cv2.imwrite(out_name, redacted)
    print(f"PNG обработан. Результат: {out_name}")


def process_pdf(filename):
    temp_dir = "temp_pdf_pages"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        pages = convert_from_path(filename, dpi=300)
    except Exception as e:
        print(f"Ошибка чтения PDF: {e}")
        return

    temp_images = []
    for idx, page in enumerate(pages):
        page_path = os.path.join(temp_dir, f"page_{idx}.png")
        page.save(page_path, "PNG")

        cv_img = cv2.imread(page_path)
        if cv_img is None:
            continue

        redacted_img = redact_image_line_by_line_with_chatgpt(cv_img)
        cv2.imwrite(page_path, redacted_img)
        temp_images.append(page_path)

    result_filename = filename.replace(".pdf", "_result.pdf")
    with open(result_filename, "wb") as f:
        f.write(img2pdf.convert(temp_images))

    print(f"PDF обработан. Результат: {result_filename}")


def main():
    while True:
        filename = input("Введите имя файла (PNG или PDF) и нажмите Enter: ").strip()
        if not filename:
            print("Имя файла пустое. Попробуйте снова.\n")
            continue

        if not os.path.exists(filename):
            print(f"Файл '{filename}' не найден. Попробуйте снова.\n")
            continue

        ext = os.path.splitext(filename)[1].lower()

        if ext == ".png":
            process_png(filename)
        elif ext == ".pdf":
            process_pdf(filename)
        else:
            print("Поддерживаются только .png и .pdf. Попробуйте снова.\n")
            continue

        break


if __name__ == "__main__":
    main()
