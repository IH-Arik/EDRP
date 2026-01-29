"""
Setup script for EDRP-DiaBD package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="edrp-diabd",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Explainable Ensemble Modeling for Diabetes Risk Prediction using the Real-World DiaBD Cohort",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IH-Arik/EDRP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
        "explainability": [
            "shap>=0.41.0",
            "lime>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "edrp-train=src.cli.train:main",
            "edrp-evaluate=src.cli.evaluate:main",
        ],
    },
)
