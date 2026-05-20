from PIL import Image
import os
import argparse

Image.MAX_IMAGE_PIXELS = None


def tiff_to_png(input_path, output_path=None):
    """
    Convert TIFF/TIF image to PNG with the same width and height.
    """

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".png"

    img = Image.open(input_path)

    print(f"Input image size: {img.size}")

    if img.mode in ["CMYK", "I;16", "I;16B", "I;16L", "I"]:
        img = img.convert("RGB")
    elif img.mode not in ["RGB", "RGBA", "L"]:
        img = img.convert("RGB")

    img.save(output_path, format="PNG")

    print(f"Saved PNG to: {output_path}")
    print(f"Output image size: {Image.open(output_path).size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TIFF/TIF image to PNG with the same size."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input TIFF/TIF file path"
    )

    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PNG file path. If not provided, save next to input file."
    )

    args = parser.parse_args()

    tiff_to_png(args.input, args.output)
