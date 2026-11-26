"""DINOv2 paired-image change detection with Hugging Face/PEFT interfaces."""

from transformers import AutoConfig, AutoImageProcessor, AutoModelForSemanticSegmentation, AutoProcessor

from .configuration_scene_change import SceneChangeConfig
from .image_processing_scene_change import SceneChangeImageProcessor, SceneChangeProcessor
from .modeling_scene_change import SceneChangeForSemanticSegmentation, SceneChangeOutput


def register_auto_classes():
    """Register local Auto classes and include source code in saved HF models."""
    AutoConfig.register(SceneChangeConfig.model_type, SceneChangeConfig, exist_ok=True)
    AutoModelForSemanticSegmentation.register(SceneChangeConfig, SceneChangeForSemanticSegmentation, exist_ok=True)
    AutoImageProcessor.register(SceneChangeConfig, slow_image_processor_class=SceneChangeImageProcessor, exist_ok=True)
    AutoProcessor.register(SceneChangeConfig, SceneChangeImageProcessor, exist_ok=True)
    SceneChangeConfig.register_for_auto_class()
    SceneChangeForSemanticSegmentation.register_for_auto_class("AutoModelForSemanticSegmentation")
    SceneChangeImageProcessor.register_for_auto_class("AutoImageProcessor")


register_auto_classes()

__all__ = [
    "SceneChangeConfig", "SceneChangeForSemanticSegmentation", "SceneChangeOutput",
    "SceneChangeImageProcessor", "SceneChangeProcessor", "register_auto_classes",
]
