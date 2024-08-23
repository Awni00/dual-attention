Getting Started
===============

Requirements
------------

- :code:`torch`
- :code:`einops`
- :code:`tiktoken` [optional] (tokenizer used for pre-trained language models)
- :code:`huggingface_hub` [optional] (for loading pre-trained model checkpoints from HF)
- :code:`safetensors` [optional] (again, for loading pre-trained model checkpoints from HF)
- :code:`gradio` [optional] (for creating graphical interfaces for visualization and running inference)
- :code:`bertviz` [optional] (for visualizing attention heads)

Installation
------------

The :code:`dual-attention` package is hosted on the `Python Package Index (PyPI) <https://pypi.org/project/dual-attention/>`_ and can be installed using :code:`pip`:

.. code-block:: bash

    pip install dual-attention

We recommend installing the package in a virtual environment.

You can also install from source by cloning the repository:

.. code-block:: bash

    git clone https://github.com/Awni00/dual-attention.git
    cd dual-attention
    pip install .

