``dual-attention`` Python Package: Dual Attention Transformers
=================================================================
.. raw:: html

   <p align='center'>
      <a href='https://www.python.org/downloads/release'>
         <img alt="Python 3.10+" src='https://img.shields.io/badge/python-3.10+-blue.svg'/>
      </a>
      <a href="https://pypi.org/project/dual-attention/">
         <img alt="PyPI" src="https://img.shields.io/pypi/v/dual-attention" alt="PyPI version">
      </a>
      <a href='https://dual-attention-transformer.readthedocs.io/en/latest/?badge=latest'>
         <img src='https://readthedocs.org/projects/dual-attention-transformer/badge/?version=latest' alt='Documentation Status' />
      </a>
      <a href=https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project>
         <img alt='PR welcome' src='https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?'/>
      </a>
      <a href=https://arxiv.org/abs/2405.16727>
         <img alt="arXiv Paper" src="https://img.shields.io/badge/arXiv-paper-brown?logo=arxiv">
      </a>
      <a href='https://github.com/Awni00/dual-attention/blob/main/LICENSE'>
         <img alt="PyPI - License" src="https://img.shields.io/pypi/l/dual-attention"/>
      </a>
   </p>

.. <code>dual-attention</code> is a Python package implementing the various components of the Dual Attention Transformer (DAT) architecture, as proposed by the paper <em><a href = https://arxiv.org/abs/2405.16727>Disentangling and Integrating Relational and Sensory Information in Transformer Architectures </a></em> by Awni Altabaa, John Lafferty.

``dual-attention`` is a Python package implementing the various components of the Dual Attention Transformer (DAT) architecture, as proposed by the paper `Disentangling and Integrating Relational and Sensory Information in Transformer Architectures <https://arxiv.org/abs/2405.16727>`_ by Awni Altabaa, John Lafferty.

   **Abstract.** The Transformer architecture processes sequences by implementing a form of neural message-passing that consists of iterative information retrieval (attention), followed by local processing (position-wise MLP). Two types of information are essential under this general computational paradigm: "sensory" information about individual objects, and "relational" information describing the relationships between objects. Standard attention naturally encodes the former, but does not explicitly encode the latter. In this paper, we present an extension of Transformers where multi-head attention is augmented with two distinct types of attention heads, each routing information of a different type. The first type is the standard attention mechanism of Transformers, which captures object-level features, while the second type is a novel attention mechanism we propose to explicitly capture relational information. The two types of attention heads each possess different inductive biases, giving the resulting architecture greater efficiency and versatility. The promise of this approach is demonstrated empirically across a range of tasks.

This documentation covers the basics of installation and usage of the ``dual-attention`` package, and provides a reference for the various components and classes implemented in the package.

High-Level Ideas of Paper
-------------------------

The Transformer architecture can be understood as an instantiation of a broader computational paradigm implementing a form of neural message-passing that iterates between two operations: 1) information retrieval (self-attention), and 2) local processing (feedforward block). To process a sequence of objects :math:`x_1, \ldots, x_n`, this general neural message-passing paradigm has the form

.. math::
   \begin{align*}
   x_i &\gets \mathrm{Aggregate}(x_i, {\{m_{j \to i}\}}_{j=1}^n)\\
   x_i &\gets \mathrm{Process}(x_i).
   \end{align*}

In the case of Transformers, the self-attention mechanism can be seen as sending messages from object :math:`j` to object :math:`i` that are encodings of the sender's features, with the message from sender :math:`j` to receiver :math:`i` given by :math:`m_{j \to i} = \phi_v(x_j)`. These messages are then aggregated according to some selection criterion based on the receiver's features, typically given by the softmax attention scores.

We posit that there are essentially two types of information that are essential under this general computational paradigm: 1) *sensory* information describing the features and attributes of individual objects, and *relational* information about the relationships between objects. The standard attention mechanism of Transformers naturally encodes the former, but does not explicitly encode the latter.

In this paper, we propose *Relational Attention* as a novel attention mechanism that enables routing of relational information between objects. We then introduce *Dual Attention*, a variant of multi-head attention combining two distinct attention mechanisms: 1) standard Self-Attention for routing sensory information, and 2) Relational Attention for routing relational information. This in turn defines an extension of the Transformer architecture with an explicit ability to reason over both types of information.

Summary of Contents of ``dual-attention`` Package
-------------------------------------------------

The ``dual-attention`` package includes the following components and features:


#. Implementations of the various core computational mechanisms of the Dual Attention Transformer architecture, including:

   - Relational Attention
   - Dual Attention
   - Symbol Assignment Mechanisms (Symbolic Attention, Position-Relative Symbols, and Positional Symbols)
   - Dual Attention Transformer Encoder and Decoder blocks

#. Implementations of Dual Attention Transformer models for various task paradigms, including:

   - Dual Attention Transformer Language Models ("Decoder-only" architecture)
   - Dual Attention Transformer Sequence-to-Sequence Encoder-Decoder Models
   - Vision Dual Attention Transformer (ViT-like architecture based on processing patches of images)

#. An interface to Huggingface Hub for loading pre-trained models (e.g., from the paper or contributed by community).

#. Utilities and applications for visualizing internal representations and running inference with trained models.


Citation
--------

For ``natbib`` or ``bibtex`` please use the following citation (as provided by Google Scholar):

.. code:: bibtex

   @article{altabaa2024disentangling,
      title={Disentangling and Integrating Relational and Sensory Information in Transformer Architectures},
      author={Awni Altabaa and John Lafferty},
      year={2024},
      journal={arXiv preprint arXiv:2402.08856}
   }

For ``biblatex``, please use the following citation (as provided by arxiv):

.. code-block:: bibtex

   @misc{altabaa2024disentangling,
      title={Disentangling and Integrating Relational and Sensory Information in Transformer Architectures},
      author={Awni Altabaa and John Lafferty},
      year={2024},
      eprint={2405.16727},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
   }

Documentation Table of Contents
-------------------------------

.. toctree::
   :maxdepth: 2

   getting_started
   layer_classes
   model_classes
   apps
   troubleshooting-contributing
   package_reference


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
