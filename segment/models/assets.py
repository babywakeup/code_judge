"""Repository-local backbone paths; never fall back to a model download."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE_ROOT = (REPO_ROOT / "../../hf_cache").resolve()


def resolve_local_backbone(source):
    """Resolve explicit paths relative to the repo, or IDs under the local cache."""
    text = str(source).strip()
    if not text:
        raise FileNotFoundError("DINOv2 backbone path is empty")
    aliases = {
        "dinov2_vits14": "facebook/dinov2-small",
        "dinov2_vitb14": "facebook/dinov2-base",
        "dinov2_vitl14": "facebook/dinov2-large",
        "dinov2_vitg14": "facebook/dinov2-giant",
    }
    text = aliases.get(text, text)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (HF_CACHE_ROOT if text.startswith("facebook/") else REPO_ROOT) / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local DINOv2 backbone directory not found: {path}")
    return path
