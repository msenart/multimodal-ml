from multimodal_ml.models.base import MultimodalModel
from multimodal_ml.models.baseline import ConcatMLP
from multimodal_ml.models.audio_gru import AudioGRU
from multimodal_ml.models.audio_mlp import AudioMLP
from multimodal_ml.models.text_mlp import TextMLP
from multimodal_ml.models.vision_mlp import VisionMLP
from multimodal_ml.models.vision_mlp_deep import VisionMLPDeep
from multimodal_ml.models.concat_fusion import ConcatFusion

MODEL_REGISTRY: dict[str, type[MultimodalModel]] = {
    "baseline": ConcatMLP,
    "audio_gru": AudioGRU,
    "audio_mlp": AudioMLP,
    "text_mlp": TextMLP,
    "vision_mlp_deep": VisionMLPDeep,
    "vision_mlp": VisionMLP,
    "concat_fusion": ConcatFusion,
}
"""Models selectable via `python -m multimodal_ml --model <name>`.

To add your own architecture: write `models/your_model.py` with a class that
subclasses `MultimodalModel` (see `models/base.py`), then add it here:

    from multimodal_ml.models.your_model import YourModel
    MODEL_REGISTRY["your_model"] = YourModel

That's the only edit needed to this file — the CLI picks it up automatically.
"""
