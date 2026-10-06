from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_WEIGHTS = PROJECT_ROOT / "yolo11s.pt"
TRAINED_WEIGHTS = PROJECT_ROOT / "weights" / "best_traffic.pt"

MODEL_CHOICES = {
    "trained": "My trained traffic model",
    "pretrained": "Pretrained YOLO11s (COCO)",
}


def default_model_choice():
    return "trained"


def resolve_model_weights(choice):
    if choice == "trained":
        if not TRAINED_WEIGHTS.is_file():
            raise FileNotFoundError(
                "The trained model is not available yet. Run "
                "'python scripts/train.py' to train it first."
            )
        return str(TRAINED_WEIGHTS)
    if choice == "pretrained":
        return str(PRETRAINED_WEIGHTS) if PRETRAINED_WEIGHTS.is_file() else "yolo11s.pt"
    raise ValueError(f"Unknown model choice: {choice}")
