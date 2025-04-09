from setuptools import setup, find_packages

setup(
    name="lunsformter",
    version="0.1.0",
    description="A lightweight, flexible transformer-like language model toolkit with inside-out generation.",
    author="Your Name",
    author_email="youremail@example.com",
    url="https://github.com/yourusername/lunsformter",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.6',
    include_package_data=True,
    license="MIT",
    keywords="transformer language-model NLP BPE inside-out generation research",
)