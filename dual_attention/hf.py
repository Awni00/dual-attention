"""
This module implements a Huggingface Hub mixin for the Dual Attention Transformer Language Model.
"""

from huggingface_hub import PyTorchModelHubMixin
from dual_attention.language_models import DualAttnTransformerLM

class DualAttnTransformerLM_HFHub(PyTorchModelHubMixin, DualAttnTransformerLM):
    """
    Huggingface Hub mixin for the Dual Attention Transformer Language Model.

    This class inherits from both the PyTorchModelHubMixin and the DualAttnTransformerLM classes.
    It allows you to load a DAT model from Huggingface Hub using the `from_pretrained` method.

    Example:
    ```python
    from dual_attention.hf import DualAttnTransformerLM_HFHub

    model = DualAttnTransformerLM_HFHub.from_pretrained("awni00/DAT-sa8-ra8-ns1024-sh8-nkvh4-343M")
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)