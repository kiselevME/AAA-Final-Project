from PIL import Image

from aiohttp.web import Response
from aiohttp.web import View

from aiohttp_jinja2 import render_template

from src.lib.utils import (open_image, get_html_avito, parse_html,
                           image_to_img_src, convert_description_to_tokens)
from src.lib.config import Variables
from src.lib.inference import get_predictions


class IndexView(View):
    template = "index.html"

    async def get(self) -> Response:
        return render_template(self.template, self.request, {})

    async def post(self) -> Response:
        try:
            form = await self.request.post()
            if form.get("image", False):
                image_path = form["image"].file
                user_description = str(form["description"])
            elif form.get("avito_url", False):
                html_text = get_html_avito(form["avito_url"])
                image_path, user_description = parse_html(html_text)
            image = open_image(image_path)

            result = get_predictions(
                image=image,
                user_description=user_description,
                request=self.request
                )

            main_image = Image.fromarray(image)
            main_image = image_to_img_src(main_image)
            ctx = {"main_image": main_image,
                   "user_description": user_description,
                   "predictions": result}
        except Exception as err:
            form = await self.request.post()
            description = str(form["description"])
            vectorizer = self.request.app["vectorizer"]
            description = convert_description_to_tokens(
                description, vectorizer["tokenizer"],
                vectorizer["encode_mapping"]
            ).to(Variables.DEVICE)
            ctx = {"error": str(err),
                   "error_type": type(err).__name__,
                   "desc": description,
                   "shape": description.shape}
        return render_template(self.template, self.request, ctx)
