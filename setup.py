from setuptools import setup, find_packages

setup(
    name="lunsft",
    version="0.1.0",
    description="Lightweight Inside-Out Byte-Pair Encoding Transformer Library",
    author="Icarogamer2441",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20"
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)