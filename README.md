# Paper Synopsis

An intelligent research paper analysis tool that uses AI vision and language models to automatically extract, analyze, and synthesize key information from research PDF documents.

## Features

- **PDF Upload & Processing**: Upload research papers and automatically parse all pages
- **Vision Analysis**: AI-powered diagram and image analysis with detailed descriptions
- **Text Understanding**: Extract key concepts, main ideas, and claims from paper text
- **Iterative Refinement**: Agentic workflow that validates and improves descriptions through feedback loops
- **Interactive UI**: Streamlit-based web interface for easy interaction
- **Context-Aware**: Analyzes images with paper context for accurate descriptions

## Tech Stack

- **LangGraph**: Agent framework for building multi-step workflows
- **LangChain**: LLM orchestration and integration
- **OpenAI API**: Vision and language models (GPT-4o-mini)
- **Streamlit**: Interactive web UI
- **PyPDF2**: PDF processing
- **NetworkX**: Graph-based workflow management

## Project Structure

```
paper_synopsis/
├── app.py                  # Streamlit web interface
├── main.py                 # Legacy implementation reference
├── pipeline/
│   ├── graph.py           # LangGraph workflow definition
│   ├── pdf_utils.py       # PDF processing utilities
│   └── schemas.py         # Pydantic models for structured outputs
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kishor-kumar-nanda/paper_synopsis.git
cd paper_synopsis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file in the root directory
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then:
1. Open your browser to `http://localhost:8501`
2. Upload a research PDF
3. Optionally enable "Manually override context" for custom analysis context
4. Click "Run Agent" to analyze the paper
5. Review AI-generated descriptions and analyses for each page

## How It Works

The application uses a multi-stage agentic workflow:

1. **Text Node**: Extracts main ideas, concepts, and claims from page text
2. **Vision Node**: Analyzes diagrams and images using GPT-4o vision capabilities
3. **Reflector Node**: Validates descriptions against paper context and provides quality feedback
4. **Synthesis Node**: Combines text and vision insights into comprehensive explanations

The workflow iteratively refines descriptions based on reflector feedback until quality targets are met.

## Key Components

### graph.py
Defines the LangGraph workflow with state management and node implementations for multi-step analysis.

### pdf_utils.py
Utilities for PDF parsing, image extraction, text extraction, and context building.

### schemas.py
Pydantic models for structured outputs including:
- `TextUnderstanding`: Extracted text concepts
- `VisionResponse`: Diagram analysis results
- `ReflectorResponse`: Quality evaluation feedback
- `SynthesizedOutput`: Final combined analysis

## Requirements

- Python 3.8+
- OpenAI API key with GPT-4o and GPT-4o-mini access
- Dependencies listed in requirements.txt

## License

MIT

## Author

Kishor Kumar Nanda
