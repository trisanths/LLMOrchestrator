# Installation Guide

## Requirements

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

## Basic Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/LLMOrchestrator.git
cd LLMOrchestrator
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development Installation

For development work, install additional dependencies:
```bash
pip install -r requirements-test.txt
```

## API Keys

1. Create a `.env` file in the project root
2. Add your API keys:
```
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key
```

## Verification

Verify installation by running tests:
```bash
pytest tests/ -v
```

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure you're in the correct virtual environment
2. **API Key Errors**: Check your `.env` file configuration
3. **Version Conflicts**: Try creating a fresh virtual environment

### Getting Help

- Open an issue on GitHub
- Check existing issues for solutions
- Review the documentation 