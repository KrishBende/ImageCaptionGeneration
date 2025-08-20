# Image Captioning Generator

This project uses a pre-trained model from the Hugging Face `transformers` library to generate descriptive captions for images.

## Setup

1. **Create and activate the virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To generate a caption for an image, run the `predict.py` script with the path to your image. Sample images are provided in the `data` directory.

For example:
```bash
python predict.py data/demo.png
```