# Characterize_GenAIModel

## Environment setup

- **Virtualenv**: Create and activate a virtual environment in the project root.
	- PowerShell: `.\.venv\Scripts\Activate.ps1`
	- Command Prompt: `.\.venv\Scripts\activate.bat`
- **Install dependencies**: `pip install -r requirements.txt`
- **Quick verification**: `python -c "import transformers, tokenizers; print(transformers.__version__, tokenizers.__version__)"`
- **Python executable (configured)**: `C:/Users/carol/Desktop/Characterize_GenAIModel/.venv/Scripts/python.exe`

If you prefer to install packages manually, run `pip install transformers tokenizers datasets torch sentencepiece accelerate` instead.