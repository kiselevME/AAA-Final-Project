import logging
from pathlib import Path

import aiohttp_jinja2
import jinja2
import aiohttp.web as web

from src.lib import views
from src.lib.model.models import create_model, create_vectorizer
from src.lib.utils import open_image
from src.lib.inference import get_predictions, predictions_postprocessing
from src.lib.config import Variables


lib = Path("src/lib")
LOGGER_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=LOGGER_FORMAT,
                    datefmt="[%H:%M:%S]",
                    level=logging.INFO)


async def recognize_renovation(request: web.Request) -> web.Response:
    try:
        post = await request.post()
        image = open_image(post.get("image").file)
        user_description = str(post.get("description"))

        predictions = get_predictions(
            image=image,
            user_description=user_description,
            request=request
        )
        if int(post.get("returnProbasFlg", 0)):
            result_json = {predict["type"]: predict["confidence"]
                           for predict in predictions}
        else:
            result_json = {"result": predictions_postprocessing(
                predictions=predictions,
                threshold=Variables.PRED_THRESHOLD
            )}
    except Exception as error:
        post = await request.post()
        result_json = {"error": error}
        request.app.logger.error(error)

    return web.json_response(result_json)


def create_app() -> web.Application:
    app = web.Application(logger=logging.getLogger())
    # setup routes
    app.router.add_static("/static/", lib / "static")
    app.router.add_view("/", views.IndexView, name="index")
    app.router.add_post("/recognizeRenovation", recognize_renovation)
    # setup templates
    aiohttp_jinja2.setup(
        app=app,
        loader=jinja2.FileSystemLoader(lib / "templates"),
    )
    app["model"] = create_model()
    app["vectorizer"] = create_vectorizer()
    return app


async def async_create_app() -> web.Application:
    app = create_app()
    return app
