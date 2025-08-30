"""
Setup script for Strat-e-GPT
Race Strategy Prediction using Machine Learning
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open(this_directory / "requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="strat-e-gpt",
    version="0.1.0",
    author="Strat-e-GPT Team",
    author_email="team@strat-e-gpt.com",
    description="Predictive Race Strategy Analysis using Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/strat-e-gpt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "strat-e-gpt=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="machine-learning, racing, strategy, prediction, data-science, python",
    project_urls={
        "Bug Reports": "https://github.com/your-username/strat-e-gpt/issues",
        "Source": "https://github.com/your-username/strat-e-gpt",
        "Documentation": "https://strat-e-gpt.readthedocs.io/",
    },
)
