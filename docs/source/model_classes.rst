Model Classes
=============

In addition to implementing the core building blocks of the DAT architecture, the ``dual-attention`` package also implements ``nn.Module`` classes for several DAT models, covering different task paradigms. In particular, the package provides the following models:

- ``DualAttnTransformerLM``: A decoder-only Dual Attention Transformer for language modeling tasks.
- ``Seq2SeqDualAttnTransformer``: An encoder-decoder Dual Attention Transformer for sequence-to-sequence tasks.
- ``VisionDualAttnTransformer``: A ViT-style Dual Attention Transformer architecture for vision tasks, expecting an image tensor as input and processing by breaking it down into patches.

These models can be instantiated and used as PyTorch modules in the same way as any other PyTorch model. 

Examples
--------

Below, we provide an example of how to instantiate and use the ``DualAttnTransformerLM`` model for language modeling tasks:

.. code-block:: python

    from dual_attention.language_models import DualAttnTransformerLM

    dat_lm = DualAttnTransformerLM(
        vocab_size=50257,      # vocabulary size
        d_model=1024,          # model dimension
        n_layers=24,           # number of layers
        n_heads_sa=8,          # number of self-attention heads
        n_heads_ra=8,          # number of relational attention headsd
        dff=None,              # feedforward intermediate dimension
        dropout_rate=0.,       # dropout rate
        activation='swiglu',   # activation function of feedforward block
        norm_first=True,       # whether to use pre-norm or post-norm
        max_block_size=1024,   # max context length
        symbol_retrieval='symbolic_attention', # type of symbol assignment mechanism
        symbol_retrieval_kwargs=dict(d_model=1024, n_heads=8, n_symbols=1024),
        pos_enc_type='RoPE'    # type of positional encoding to use
    )

    idx = torch.randint(0, 50257, (1, 1024+1))
    x, y = idx[:, :-1], idx[:, 1:]
    logits, loss = dat_lm(x, y)
    logits # shape: (1, 1024, 50257)

Below, we provide an example of how to instantiate and use the ``VisionDualAttnTransformer`` model for vision tasks:

.. code-block:: python

    from dual_attention.vision_models import VisionDualAttnTransformer

    img_shape = (3, 224, 224)
    patch_size = (16, 16)
    n_patches = (img_shape[1] // patch_size[0]) * (img_shape[2] // patch_size[1])

    dat_vision = VisionDualAttnTransformer(
        image_shape=img_shape,     # shape of input image
        patch_size=patch_size,     # size of patch
        num_classes=1000,          # number of classes
        d_model=512,               # model dimension
        n_layers=6,                # number of layers
        n_heads_sa=4,              # number of self-attention heads
        n_heads_ra=4,              # number of relational attention heads
        dff=2048,                  # feedforward intermediate dimension
        dropout_rate=0.1,          # dropout rate
        activation='swiglu',       # activation function of feedforward block
        norm_first=True,           # whether to use pre-norm or post-norm
        symbol_retrieval='position_relative', # type of symbol assignment mechanism
        symbol_retrieval_kwargs=dict(symbol_dim=512, max_rel_pos=n_patches+1),
        ra_kwargs=dict(symmetric_rels=True, use_relative_positional_symbols=True),
        pool='cls',                # type of pooling (class token)
    )

    img = torch.randn(1, *img_shape)
    logits = dat_vision(img)
    logits.shape # shape: (1, 1000)

Loading Pre-trained Models from Hugging Face
--------------------------------------------

The ``dual_attention.hf`` module provides a convenient interface to Huggingface Hub for loading pre-trained DAT models or sharing your own trained models.

To load a pre-trained model from Hugging Face, you can simply run the following code:

.. code:: python

    from dual_attention.hf import DualAttnTransformerLM_HFHub

    model = DualAttnTransformerLM_HFHub.from_pretrained("awni00/DAT-sa16-ra16-nr128-ns2048-sh16-nkvh8-1.27B")
