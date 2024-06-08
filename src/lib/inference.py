import numpy as np
import aiohttp.web as web

from src.lib.utils import preprocess_image, convert_description_to_tokens
from src.lib.config import Variables


def get_predictions(
    image: np.array,
    user_description: str,
    request: web.Request
) -> tuple:
    image = preprocess_image(image)["image"].to(Variables.DEVICE)

    vectorizer = request.app["vectorizer"]
    description = convert_description_to_tokens(
        user_description, vectorizer["tokenizer"],
        vectorizer["encode_mapping"]
    ).to(Variables.DEVICE)
    model = request.app["model"]
    model.eval()
    predictions = model(image.unsqueeze(0), description.unsqueeze(0))\
        .softmax(dim=1).detach().cpu()[0]
    result = []
    for i in range(len(predictions)):
        result.append({
            "type": Variables.TARGETS[i],
            "confidence": predictions[i].item(),
        })

    result = sorted(result, key=lambda x: x["confidence"], reverse=True)
    return result


def predictions_postprocessing(predictions: list, threshold: float) -> str:
    # пользуемся отсортированностью предсказаний
    if predictions[0]["confidence"] < threshold:
        return "Невозможно определить"
    return predictions[0]["type"]
