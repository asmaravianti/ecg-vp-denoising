from setuptools import setup, find_packages

setup(
    name="ecgdae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0",
        "torchaudio>=2.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "einops>=0.7.0",
        "PyYAML>=6.0.0",
        "rich>=13.7.0",
        "wfdb>=4.1.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
        "PyWavelets>=1.6.0",
    ],
)

