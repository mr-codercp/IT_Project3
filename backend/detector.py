import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "dima806/ai_vs_real_image_detection"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


print("Loading AI detection model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("AI detection model loaded.")


def detect_ai_probability(frame_rgb):
    """
    Takes an RGB numpy frame and returns:
    AI probability between 0 and 1.
    """

    image = Image.fromarray(frame_rgb)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    ai_index = None

    for index, label in model.config.id2label.items():

        label = label.upper()

        if label in ["FAKE", "AI", "AI-GENERATED", "GENERATED"]:

            ai_index = int(index)
            break

    if ai_index is None:
        raise RuntimeError(
            f"Could not identify AI label. "
            f"Available labels: {model.config.id2label}"
        )

    ai_probability = probabilities[ai_index].item()

    return float(ai_probability)
