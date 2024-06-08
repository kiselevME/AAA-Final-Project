from io import BufferedReader, BytesIO
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.lib.config import Variables
from base64 import b64encode
import pymorphy3


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
