# Sources and licenses

The shared DINOv2 encoder / paired attention / convolutional decoder architecture,
post-norm attention equations, feature facets and legacy weight names are adapted
from **Robust Scene Change Detection Using Visual Foundation Models and
Cross-Attention Mechanisms**, Copyright (c) 2024 Chad Lin, MIT License:
https://github.com/ChadLin9596/Robust-Scene-Change-Detection

The accompanying LICENSE preserves that notice. No source from its CoAttention,
C-3PO, or DR-TANet ablation modules is included here.

The PSCD fixed train/validation/test split IDs and aligned/shifted crop protocol
in `data/` are also adapted from the same MIT-licensed project. The optional
high-resolution panorama resize is a local addition, not an upstream protocol.
No PSCD images are included or redistributed. PSCD dataset terms independently
restrict use to academic research and require permission for redistribution:
https://kensakurada.github.io/pscd/term_of_use.html .

The backbone is provided by the installed Hugging Face Transformers library
(Apache-2.0), rather than copied or loaded with torch.hub. DINOv2 model weights
retain the licenses and terms of their respective model cards, e.g.
https://huggingface.co/facebook/dinov2-small .