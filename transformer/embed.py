from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformer.encoder import MultiScaleEncoder

FINETUNED_PATH = Path(__file__).parent / "finetuned.pt"


def load_encoder(
    path: Path = FINETUNED_PATH,
    device: str = "cpu",
) -> "MultiScaleEncoder | None":
    """
    Load fine-tuned MultiScaleEncoder from disk.
    Returns None if the checkpoint does not exist — callers degrade to bare features.
    """
    from transformer.encoder import MultiScaleEncoder
    import torch

    path = Path(path)
    if not path.exists():
        return None
    model = MultiScaleEncoder()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model
