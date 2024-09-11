Layer Classes
=============

The Dual Attention Transformer (DAT) architecture is an extension of the Transformer framework which introduces several new computational mechanisms. The ``dual-attention`` package implements these new mechanisms as PyTorch ``nn.Module`` classes. We recommend reading the `paper <https://arxiv.org/abs/2405.16727>`_ first to better understand the package and documentation. In this section, we highlight some key classes and their usage.

Relational Attention
--------------------

Relational attention is an attentional mechanism proposed as part of the DAT architecture which routes relational information between elements of the input. It takes the following form:

.. math::

    \begin{align*}
        \text{RelationalAttention}(x, (y_1, \ldots, y_n)) &= \sum_{i} \alpha_{i}(x, \boldsymbol{y}) (r(x, y_i) W_r + s_i W_s), \\
        \alpha(x, \boldsymbol{y}) &= \text{softmax}([\langle \phi_q^{\mathrm{attn}}(x), \phi_k^{\mathrm{attn}}(y_i) \rangle]_{i=1}^{n}), \\
        r(x, y) &= (\langle \phi_{q,\ell}^{\mathrm{rel}}(x),  \phi_{k, \ell}^{\mathrm{rel}}(y))_{\ell \in [d_r]}, \\
        (s_1, \ldots, s_n) &= \text{SymbolRetriever}(\boldsymbol{y}; S_{\mathrm{lib}}),
    \end{align*}


The ``RelationalAttention`` class in the ``dual_attention.relational_attention`` module implements the relational attention mechanism. It can be instantiated and used as a PyTorch layer as follows:

.. code-block:: python

    from dual_attention.relational_attention import RelationalAttention

    relational_attn = RelationalAttention(
        d_model=64,                # model dimension
        n_heads=4,                 # number of heads
        n_relations=16,            # number of relations
        dropout=0.0,               # dropout rate
        key_dim=None,              # key dimension
        n_kv_heads=None,           # number of key-value heads (e.g., for MQA or GQA)
        rel_activation='identity', # activation function applied to relations
        rel_proj_dim=None,         # relation projection dimension (analogous to key_dim)
        add_bias_kv=False,         # whether to add bias to key-value projection
        add_bias_out=False,        # whether to add bias to output projection
        total_n_heads=None,        # total number of heads of DualAttn (if used within DualAttn)
        symmetric_rels=False,      # whether relations should be constrained to be symmetric
        use_relative_positional_symbols=False # whether to use relative positional symbols
        )
    x = torch.randn(1, 10, 64)
    s = torch.randn(1, 10, 64)
    out, *_ = relational_attn(x, s)

    print(f'x: {x.shape}, s: {s.shape}, out: {out.shape}')
    # x: torch.Size([1, 10, 64]), s: torch.Size([1, 10, 64]), out: torch.Size([1, 10, 64])


Symbol Assignment Mechanisms
----------------------------

The ``dual_attention.symbol_retrieval`` module implements several symbol assignment mechanisms which can be used to assign symbols to elements of the input in relational attention. The following symbol assignment mechanisms are implemented:

- ``SymbolicAttention``: Assigns symbols based on an attention operation to a library of symbols, learned as parameters of the model. This corresponds to computing a type of soft "equivalence class" over the input elements.
- ``PositionalSymbolRetriever``: Elements of the input are assigned a symbol based on their position in the input sequence (essentially a positional embedding).
- ``PositionRelativeSymbolRetriever``: Elements of the input are assigned different symbols depending on the "receiver", encoding the *relative* of the sender with respect to the receiver. To use this, the ``use_relative_positional_symbols`` argument of the ``RelationalAttention`` module must be set to ``True``.

Dual Attention
--------------

Dual attention is a variant of multi-head attention in which there exists two sets of attention heads of different types: standard self-attention and relational attention. The two sets of attention heads are computed in parallel and concatenated to form the final output. This equips the model with two versatile computational mechanisms for processing and attending over both sensory and relational information.

The ``DualAttention`` class in the ``dual_attention.dual_attention`` module implements the dual attention mechanism. It can be instantiated and used as a PyTorch layer as follows:

.. code-block:: python

    from dual_attention.dual_attention import DualAttention

    dual_attention = DualAttention(
        d_model=64,                # model dimension
        n_heads_sa=4,              # number of self-attention heads
        n_heads_ra=4,              # number of relational attention heads
        dropout=0.,                # dropout rate
        sa_kwargs=None,            # extra kwargs for self-attention
        ra_kwargs=None,            # extra kwargs for relational attention
        share_attn_params=False,   # whether to share attention parameters between self-attention and relational attention
        ra_type='relational_attention', # type of relational attention to use; used in ablation experiments in paper
        )

    x = torch.randn(1, 10, 64)
    s = torch.randn(1, 10, 64)
    out, *_ = dual_attention(x, s)

    print(f'x: {x.shape}, s: {s.shape}, out: {out.shape}')
    # x: torch.Size([1, 10, 64]), s: torch.Size([1, 10, 64]), out: torch.Size([1, 10, 64])

Dual Attention Transformer Blocks
---------------------------------

The blocks in a Dual Attention Transformer architecture mirror those of a standard Transformer, but with dual attention replacing standard multi-head self-attention. They consist of dual-attention followed by a feedforward block, with layer normalization and residual connections around each sub-block.

The ``DualAttentionEncoderBlock`` and ``DualAttentDecoderBlock`` classes in the ``dual_attention.dual_attn_blocks`` module implement Dual Attention Transformer blocks. They can be instantiated and used as PyTorch layers as follows:

.. code-block:: python

    from dual_attention.dual_attn_blocks import DualAttnEncoderBlock

    dat_block = DualAttnEncoderBlock(
        d_model=64,                # model dimension
        n_heads_sa=4,              # number of self-attention heads
        n_heads_ra=4,              # number of relational attention heads
        dff=256,                   # feedforward intermediate dimension
        activation='gelu',         # activation function of feedforward block
        dropout_rate=0.,           # dropout rate
        norm_first=True,           # whether to use pre-norm or post-norm
        norm_type='layernorm',     # type of normalization to use (e.g., 'layernorm' or 'rmsnorm')
        sa_kwargs=None,            # extra kwargs for self-attention
        ra_kwargs=None,            # extra kwargs for relational attention
        share_attn_params=False,   # whether to share attention parameters between self-attention and relational attention
        ra_type='relational_attention', # type of relational attention to use; used in ablation experiments in paper
        bias=True,                 # whether to use bias
        causal=False               # whether to use causal masking (e.g., for use in language model)
        )

    x = torch.randn(1, 10, 64)
    s = torch.randn(1, 10, 64)
    out, *_ = dat_block(x, s)

    print(f'x: {x.shape}, s: {s.shape}, out: {out.shape}')
    # x: torch.Size([1, 10, 64]), s: torch.Size([1, 10, 64]), out: torch.Size([1, 10, 64])
