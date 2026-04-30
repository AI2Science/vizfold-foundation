from PIL import Image
import requests


def load_image(url):
    image = Image.open(requests.get(url, stream=True).raw)
    return image