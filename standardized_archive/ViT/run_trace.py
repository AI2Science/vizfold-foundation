from utils import load_image
from trace_vit import trace_vit


IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def main():

    image = load_image(IMAGE_URL)

    archive = trace_vit(
        image=image,
        archive_path="vit_trace.zarr"
    )

    print("Trace archive created at vit_trace.zarr")


if __name__ == "__main__":
    main()