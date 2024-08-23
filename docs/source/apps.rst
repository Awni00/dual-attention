Apps for Visualization and Running Inference
============================================

The ``dual_attention.model_analysis`` subpackage provides two apps for visualizing the internal representations of a trained DAT language model or running inference in a graphical user interface. The apps are also hosted online on Huggingface Spaces.

To run the inference app, you can simply run the following code:

.. code-block:: bash

    python -m dual_attention.model_analysis.lm_inference_app

.. raw:: html

    <iframe
	src="https://awni00-dat-lm-inference.hf.space"
	frameborder="0"
	width="100%"
    height="600"
    ></iframe>


To run the visualization app, you can simply run the following code:

.. code-block:: bash

    python -m dual_attention.model_analysis.lm_visualization_app

.. raw:: html

    <iframe
        src="https://awni00-dat-lm-visualization.hf.space"
        frameborder="0"
        width="100%"
        height="600"
    ></iframe>
