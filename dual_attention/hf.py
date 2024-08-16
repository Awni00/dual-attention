from huggingface_hub import PyTorchModelHubMixin
from dual_attention.language_models import DualAttnTransformerLM

class DualAttnTransformerLM_HFHub(PyTorchModelHubMixin, DualAttnTransformerLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)