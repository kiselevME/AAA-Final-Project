import os
from io import BufferedReader, BytesIO
from PIL import Image
from base64 import b64encode
import requests

import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pymorphy3
from src.lib.config import Variables


def open_image(image_link: BufferedReader) -> np.array:
    image = Image.open(image_link)
    image_array = np.asarray(image)
    if image.mode == "RGBA":
        # для обработки png
        image_array = image_array[:, :, :-1]
    return image_array


def preprocess_image(image: np.array) -> torch.tensor:
    test_transform = A.Compose(
        [
            A.Resize(224, 224, p=1),
            A.Normalize(mean=np.array(Variables.MEAN),
                        std=np.array(Variables.STD)),
            ToTensorV2(),
        ]
    )
    return test_transform(image=image)


def image_b64encode(image: Image) -> str:
    with BytesIO() as io:
        image.save(io, format="png", quality=100)
        io.seek(0)
        return b64encode(io.read()).decode()


def image_to_img_src(image: Image) -> str:
    return f"data:image/png;base64,{image_b64encode(image)}"


def convert_description_to_tokens(
    text: str,
    tokenizer,
    encode_mapping
) -> np.array:
    morph = pymorphy3.MorphAnalyzer()
    text_tokens = tokenizer(text)
    text_tokens_normal_form = [morph.parse(token)[0].normal_form
                               for token in text_tokens]
    text_tokens_enc = [encode_mapping.get(token, Variables.NOT_IN_VOCAB)
                       for token in text_tokens_normal_form]
    return torch.tensor(text_tokens_enc).long()


def get_html_avito(avito_url: str):
    # r = requests.get(avito_url) ## локально не работает, поэтому закомментил
    # html_text = r.text          ## локально не работает, поэтому закомментил

    html_data = pd.read_csv("examples/urls_texts.csv")
    if avito_url in set(html_data["url"]):
        html_text = html_data.loc[html_data["url"] == avito_url, "text"].item()
    else:
        raise KeyError("Введен недопустимый url")
    return html_text


def parse_html(html_text: str) -> tuple[str]:
    title = parse_title(html_text)

    image_url = parse_image_url(html_text)
    image_path = download_image(image_url)

    description = parse_description(html_text)
    return title, image_path, description


def parse_title(html_text: str) -> str:
    req_fragment_marker = "item-view/title-info"
    fragment = "".join(
        [element for element in html_text.split("<div")
         if req_fragment_marker in element])

    title_start_marker = 'data-marker="item-view/title-info">'
    pos = fragment.index(title_start_marker)
    title = fragment[pos + len(title_start_marker):]

    replace_words = ["</h1>", "</div>"]
    for word in replace_words:
        title = title.replace(word, "")

    return title


def parse_image_url(html_text: str) -> str:
    # кусок, по которому выделяем нужный фрагмент с картинкой
    req_fragment_marker = "image-frame-wrapper-_NvbY"
    # кусок, указывающий, что дальше будет ссылка
    req_url_marker = "data-url="

    fragment = "".join([element for element in html_text.split("<div")
                        if req_fragment_marker in element])
    urls = [item.replace(req_url_marker, "").replace('"', "")
            for item in fragment.split(" ") if req_url_marker in item]
    if len(urls) != 1:
        raise ValueError("Количество найденных ссылок с картинкой: "
                         f"{len(urls)} (ожидается только одна)")

    return urls[0]


def download_image(image_url: str) -> str:
    image_nm = image_url.split("/")[-1]

    base_path = "data/images"
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    image_path = f"{base_path}/{image_nm}"

    # тут можно добавить ретраи (для mvp оставил так)
    r = requests.get(image_url, stream=True, timeout=1)
    if r.status_code == 200:
        with open(image_path, "wb") as f:
            for chunk in r:
                f.write(chunk)
    else:
        raise ConnectionError("Не удалось загрузить изображение "
                              f"(код ошибки: {r.status_code})")
    return image_path


def parse_description(html_text: str) -> str:
    req_fragment_marker = "item-view/item-description"
    fragment = "".join(
        [element.replace(req_fragment_marker, "")
            for element in html_text.split("<div")
            if req_fragment_marker in element])

    desc_start_marker = 'itemProp="description">'
    pos = fragment.index(desc_start_marker)
    desc = fragment[pos + len(desc_start_marker):]
    desc = desc.replace("<br>", "\n")
    desc = desc.replace("<p>", "\n")

    replace_words = ["<div>", "</div>", "<br>", "</p>"]
    for word in replace_words:
        desc = desc.replace(word, "")

    return desc
