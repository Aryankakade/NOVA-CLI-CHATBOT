from setuptools import setup, find_packages
import os

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "🚀 Advanced AI-Powered CLI Assistant with Professional Agents"

# Read requirements
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    # Fallback requirements based on your CLI imports
    requirements = [
        "textual>=0.40.0",
        "rich>=13.0.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "colorama>=0.4.4",
        "pyttsx3>=2.90",
        "SpeechRecognition>=3.10.0",
        "azure-cognitiveservices-speech>=1.24.0",
        "Pillow>=9.0.0",
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "openpyxl>=3.0.9",
        "chromadb>=0.4.0",
        "langchain-community>=0.0.20",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "transformers>=4.21.0",
    ]

setup(
    name="nova-cli-ai",
    version="1.0.0",
    author="Aryan Kakade",
    author_email="your.aryankakade143@gmail.com",  # Replace with your actual email
    description="🚀 Advanced AI-Powered CLI Assistant with Professional Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aryankakade/NOVA-CLI",
    packages=find_packages(),
    py_modules=[],  # Will auto-detect all .py files in nova_cli/
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Shells",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nova-cli=nova_cli.main:main",
            "nova=nova_cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "nova_cli": [
            "*.py",  # Include all your Python files
            "src/**/*",
            "ML/**/*",
            "agents/**/*",
            "templates/*",
            "data/*",
        ],
    },
    keywords="ai cli assistant chatbot agents voice textual professional coding business medical",
    project_urls={
        "Bug Reports": "https://github.com/Aryankakade/NOVA-CLI/issues",
        "Source": "https://github.com/Aryankakade/NOVA-CLI",
        "Documentation": "https://github.com/Aryankakade/NOVA-CLI#readme",
    },
)