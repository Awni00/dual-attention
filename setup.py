import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dual_attention", # module name
    version="0.0.2", # version for this release

    # author information
    author="Awni Altabaa",
    author_email="awni.altabaa@yale.edu",

    # short description
    description="""Python package implementing the Dual Attention Transformer (DAT), as proposed by the paper "Disentangling and Integrating Relational and Sensory Information in Transformer Architectures" by Awni Altabaa, John Lafferty.""",

    long_description=long_description,
    long_description_content_type="text/markdown",

    # project url
    project_urls={
        'Documentation': 'https://dual-attention-transformer.readthedocs.io/',
        'Source':'https://github.com/Awni00/dual-attention',
        'Tracker':'https://github.com/Awni00/dual-attention/issues'},

    packages=setuptools.find_packages(),

    install_requires=[
        'einops'
       ],
    extra_require={
        'all': ['tiktoken', 'huggingface_hub', 'safetensors']
    },

    license="MIT",

    # classifiers like program is suitable for python3, just leave as it is.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)