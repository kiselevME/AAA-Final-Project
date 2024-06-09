import os
import datetime
import numpy as np
import pandas as pd
import asyncio
import aiohttp

N_REQUESTS = 1000


async def make_post_request(
    session: aiohttp.ClientSession,
    url: str,
    files: dict
):
    async with session.post(url, data=files) as response:
        if response.status == 200:
            return response.status


async def run_test(n: int, handler_inputs: list[dict]):
    url = "http://0.0.0.0:8000/recognizeRenovation"
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(
            make_post_request(session, url, handler_inputs[i])
            for i in range(n)
        ))
    return results


if __name__ == "__main__":
    images_folder = "/Users/max/Desktop/AAA project/images_1"
    dataset_path = "/Users/max/Desktop/AAA project/data.csv"

    # подготавливаю данные
    data = pd.read_csv(dataset_path)
    images_id_folder = [f.replace(".jpg", "")
                        for f in os.listdir(images_folder)
                        if os.path.isfile(os.path.join(images_folder, f))]
    data = data[data["image_id"].astype(str).isin(images_id_folder)]
    data = data.sample(min(N_REQUESTS, data.shape[0]))

    handler_inputs = []
    for image_id, desc in data[["image_id", "description"]].values:
        image_file = open(f"{images_folder}/{image_id}.jpg", "rb").read()
        binary_image = bytearray(image_file)
        handler_inputs.append(
            {"image": binary_image, "description": desc}
        )
    print("Данные подготовлены.")

    time_start = datetime.datetime.now()
    results = asyncio.run(run_test(n=N_REQUESTS,
                                   handler_inputs=handler_inputs))
    time_finish = datetime.datetime.now()

    total_time_s = (time_finish - time_start).seconds

    avg_success = np.mean(np.array(results) == 200)
    print(f"Доля успешных запросов: {avg_success:.3f}")
    print(f"Суммарное время {N_REQUESTS} запрсов: {total_time_s} сек.")
    print(f"avg rps: {N_REQUESTS / total_time_s:.3f}")
    # на обычном mac air m1 получается ~ 4 rps
