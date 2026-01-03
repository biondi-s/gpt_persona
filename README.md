# GPT Persona Survey Generator

Toolkit for generating synthetic survey responses using OpenAI. Personas are sampled from weighted category distributions (`setup_data/demo_json`) and used to answer survey questions defined in `setup_data/survey_json`. The notebooks orchestrate prompt creation, call the `gpt-5-mini` chat model, and collect responses for downstream analysis.

## Project Structure
- `utils.py` core helpers to sample personas, build system/QA prompts, and query the OpenAI API.
- `generate_answers.ipynb` end-to-end run: load configs, generate personas, query the model, and save results.
- `analysis_answers.ipynb` light analysis of generated results (distribution checks, basic stats).
- `setup_data/demo_json` persona category definitions with probabilities.
- `setup_data/survey_json` survey questions, possible answers, and target/ground-truth metadata.
- `output_data/json_answers` example generated answers (written by `generate_answers.ipynb`).

## Prerequisites
- Python 3.10+ (project developed with Python 3.12).
- OpenAI API access and key.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

## Usage
1) Open and run `generate_answers.ipynb` to create personas, send survey prompts to the model, and write `output_data/json_answers`. Update the `OPENAI_API_KEY` cell or rely on the environment variable above.  
2) Open and run `analysis_answers.ipynb` to validate sampling distributions and compute simple aggregates of the generated answers.  
3) Adjust `setup_data/demo_json` (persona priors) and `setup_data/survey_json` (questions/answers) to explore different scenarios; rerun the notebook after edits.

## Notes
- The `output_data` folder is ignored by default for new artifacts; commit only representative samples if desired.
- `utils.query_gpt5_mini` assumes the `gpt-5-mini` model is available to your account; swap the model name if needed.
