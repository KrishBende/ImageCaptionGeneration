
import argparse
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

def generate_caption(image_path):
    """
    Generates a caption for a given image.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        return "Error: Image not found at the specified path."

    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # prepare image for the model
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a caption for an image.")
    parser.add_argument("image_path", type=str, help="The path to the image file.")
    args = parser.parse_args()

    caption = generate_caption(args.image_path)
    print(caption)
