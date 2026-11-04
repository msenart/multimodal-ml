from multimodal_ml.models.base import MultimodalModel
from multimodal_ml.models.baseline import ConcatMLP

MODEL_REGISTRY: dict[str, type[MultimodalModel]] = {
    "baseline": ConcatMLP,
}
"""Models selectable via `python -m multimodal_ml --model <name>`.

To add your own architecture: write `models/your_model.py` with a class that
subclasses `MultimodalModel` (see `models/base.py`), then add it here:

    from multimodal_ml.models.your_model import YourModel
    MODEL_REGISTRY["your_model"] = YourModel

That's the only edit needed to this file — the CLI picks it up automatically.
"""
