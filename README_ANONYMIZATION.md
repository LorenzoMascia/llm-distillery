# Data Anonymization Project with 1B LLM

This project uses LLM Distillery to create a 1B parameter model specialized in anonymizing sensitive data (PII - Personally Identifiable Information).

## Objective

Given input text, the model returns a JSON with:
- `anonymized_text`: text with all sensitive information replaced
- `replaced_tokens`: array of objects with `replaced_value` (placeholder) and `original_value` (original value)

## Example

**Input:**
```
My name is Mario Rossi and my email is mario.rossi@gmail.com. You can call me at +39 333 1234567.
```

**Output:**
```json
{
  "anonymized_text": "My name is [NAME_1] and my email is [EMAIL_1]. You can call me at [PHONE_1].",
  "replaced_tokens": [
    {
      "replaced_value": "[NAME_1]",
      "original_value": "Mario Rossi"
    },
    {
      "replaced_value": "[EMAIL_1]",
      "original_value": "mario.rossi@gmail.com"
    },
    {
      "replaced_value": "[PHONE_1]",
      "original_value": "+39 333 1234567"
    }
  ]
}
```

## Project File Structure

```
llm-distillery/
├── config/
│   ├── prompts_anonymization.yaml           # Prompt configuration for anonymization
│   └── training_config_1b_anonymization.yaml # Training config for 1B model
├── data/
│   └── examples/
│       └── anonymization_samples.jsonl       # Sample dataset (10 samples)
├── scripts/
│   ├── generate_anonymization_dataset.py     # Script to generate dataset with OpenAI
│   ├── train_anonymization_model.py          # Script for training
│   └── test_anonymization.py                 # Script to test the model
├── .env                                       # Environment configuration
└── README_ANONYMIZATION.md                    # This guide
```

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Edit the `.env` file and insert your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

## Complete Workflow

### Step 1: Generate Training Dataset

Use OpenAI GPT-4 as "teacher model" to generate anonymization examples:

```bash
python scripts/generate_anonymization_dataset.py \
    --config config/prompts_anonymization.yaml \
    --output data/processed/anonymization_training_data.jsonl \
    --num-samples 5000 \
    --model gpt-4-turbo-preview
```

**Parameters:**
- `--num-samples`: number of examples to generate (recommended: 5000-10000)
- `--model`: OpenAI model to use as teacher
- `--verbose`: enable detailed logging

**Notes:**
- Generating 5000 examples may take 2-4 hours and cost approximately $50-100 in API credits
- For testing, start with 100 examples: `--num-samples 100`

### Step 2: Model Fine-Tuning

Train the TinyLlama 1.1B model with the generated dataset:

```bash
python scripts/train_anonymization_model.py \
    --dataset data/processed/anonymization_training_data.jsonl \
    --config config/training_config_1b_anonymization.yaml \
    --output-dir models/anonymization_1b
```

**Hardware Requirements:**
- GPU with at least 8GB VRAM (RTX 3060, RTX 4060, or better)
- 16GB system RAM
- 10GB disk space

**Training Time:**
- With 5000 examples: approximately 1-2 hours on RTX 3090
- With 10000 examples: approximately 3-4 hours

**Parameters:**
- `--resume-from-checkpoint`: resume training from checkpoint
- `--verbose`: detailed logging

### Step 3: Model Testing

Test the trained model on new texts:

#### Interactive Mode
```bash
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --interactive
```

#### Single Test
```bash
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --text "My name is Giovanni Rossi, email: g.rossi@email.it"
```

#### Test from File
```bash
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --file data/test_input.txt
```

#### Test with Predefined Examples
```bash
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Advanced Configuration

### Change Base Model

If you want to use a different model than TinyLlama 1.1B, modify `config/training_config_1b_anonymization.yaml`:

```yaml
model:
  base_model: "microsoft/phi-2"  # Phi-2 (2.7B)
  # or
  base_model: "Qwen/Qwen-1_8B"   # Qwen 1.8B
```

### Customize Prompts

Modify `config/prompts_anonymization.yaml` to:
- Add new types of sensitive data
- Change placeholder format
- Add domain-specific examples

### Training Optimization

Modify `config/training_config_1b_anonymization.yaml`:

```yaml
training:
  num_train_epochs: 5              # More epochs = better learning
  per_device_train_batch_size: 8   # Increase if you have more VRAM
  learning_rate: 3.0e-4            # Adjust for better convergence
```

## Supported PII Types

The model is trained to recognize and anonymize:

- 👤 **Names** (individuals)
- 📧 **Emails**
- 📱 **Phone numbers** (various international formats)
- 🏠 **Addresses** (complete with street, city, ZIP)
- 📅 **Dates** (birth dates, appointments)
- 🆔 **Tax IDs** (Italy: Codice Fiscale)
- 💳 **Credit card numbers**
- 🔢 **SSN** (Social Security Number - USA)
- 🏦 **IBAN**
- 📋 **VAT numbers** (Italy: P.IVA)
- 🏥 **Patient ID / Insurance ID**
- 👔 **Employee ID**

## Troubleshooting

### Error: Out of Memory during training

Reduce batch size in `training_config_1b_anonymization.yaml`:
```yaml
per_device_train_batch_size: 4  # from 8 to 4
gradient_accumulation_steps: 4   # from 2 to 4
```

### Error: OpenAI API Rate Limit

Add delays between requests by modifying the generator, or use a smaller number of samples.

### Model doesn't generate valid JSON

- Increase the number of examples in the dataset (from 5000 to 10000)
- Increase the number of epochs (from 5 to 8)
- Reduce temperature in inference (already set to 0.1)

### Model doesn't recognize some PII types

Add more specific examples in `prompts_anonymization.yaml` in the examples section.

## Expected Performance

With a well-trained model (5000+ examples, 5 epochs):

- **Accuracy**: ~95% in recognizing common PII
- **Speed**: ~50 tokens/sec on RTX 3090
- **Latency**: 1-3 seconds for average text (100-200 words)
- **Model size**: ~2.5GB (with LoRA adapters)

## Production Use

### Python Integration

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, "models/anonymization_1b")
tokenizer = AutoTokenizer.from_pretrained("models/anonymization_1b")

# Anonymization function
def anonymize(text: str) -> dict:
    prompt = f"""### Instruction:
Analyze the following text and anonymize all personally identifiable information (PII). Return a JSON object with the anonymized text and all replaced tokens.

### Input:
{text}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].strip()
    return json.loads(response)

# Use
result = anonymize("Mario Rossi, email: m.rossi@example.com")
print(result["anonymized_text"])
```

### REST API (FastAPI example)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AnonymizationRequest(BaseModel):
    text: str

class AnonymizationResponse(BaseModel):
    anonymized_text: str
    replaced_tokens: list

@app.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_endpoint(request: AnonymizationRequest):
    result = anonymize(request.text)
    return result
```

## Batch Processing

For processing multiple texts efficiently:

```python
def batch_anonymize(texts: list[str], batch_size: int = 4) -> list[dict]:
    """
    Anonymize multiple texts in batches.

    Args:
        texts: List of texts to anonymize
        batch_size: Number of texts to process at once

    Returns:
        List of anonymization results
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = [anonymize(text) for text in batch]
        results.extend(batch_results)

    return results

# Example usage
texts = [
    "Mario Rossi, email: m.rossi@example.com",
    "Contact: John Doe, phone: +1-555-123-4567",
    "Patient: Jane Smith, SSN: 123-45-6789"
]

results = batch_anonymize(texts)
for i, result in enumerate(results):
    print(f"Text {i+1}:")
    print(f"  Anonymized: {result['anonymized_text']}")
    print(f"  Replaced: {len(result['replaced_tokens'])} tokens\n")
```

## De-anonymization (Optional)

If you need to reverse the anonymization:

```python
def deanonymize(anonymized_text: str, replaced_tokens: list[dict]) -> str:
    """
    Restore original values in anonymized text.

    Args:
        anonymized_text: Text with placeholders
        replaced_tokens: List of replacement mappings

    Returns:
        Original text with sensitive data restored
    """
    result = anonymized_text

    # Sort by replaced_value length (descending) to avoid partial replacements
    tokens = sorted(replaced_tokens, key=lambda x: len(x['replaced_value']), reverse=True)

    for token in tokens:
        result = result.replace(token['replaced_value'], token['original_value'])

    return result

# Example
anonymized = "My name is [NAME_1] and my email is [EMAIL_1]."
tokens = [
    {"replaced_value": "[NAME_1]", "original_value": "Mario Rossi"},
    {"replaced_value": "[EMAIL_1]", "original_value": "mario.rossi@gmail.com"}
]

original = deanonymize(anonymized, tokens)
print(original)  # "My name is Mario Rossi and my email is mario.rossi@gmail.com."
```

## Docker Deployment

Create a Dockerfile for production deployment:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY models/anonymization_1b /app/models/anonymization_1b
COPY src /app/src

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring and Logging

Add logging for production use:

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anonymization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def anonymize_with_logging(text: str) -> dict:
    """Anonymize text with logging."""
    start_time = datetime.now()

    try:
        logger.info(f"Processing text (length: {len(text)})")
        result = anonymize(text)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Anonymization completed in {duration:.2f}s, "
                   f"replaced {len(result['replaced_tokens'])} tokens")

        return result

    except Exception as e:
        logger.error(f"Anonymization failed: {str(e)}")
        raise
```

## License

MIT License - see LICENSE file

## Support

For issues or questions:
- Open an issue on GitHub
- Check the main documentation in [README.md](README.md)

## References

- [LLM Distillery](https://github.com/LorenzoMascia/llm-distillery)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- [HOWAI - Knowledge Distillation](https://www.howai.cloud/fine-tuning-knowledge-distillation-llm.html)

## Citation

If you use this project in your research or production, please cite:

```bibtex
@misc{llm-distillery-anonymization,
  title={LLM Distillery: Data Anonymization with 1B Parameter Models},
  author={Mascia, Lorenzo},
  year={2025},
  publisher={GitHub},
  url={https://github.com/LorenzoMascia/llm-distillery}
}
```

## Changelog

### Version 1.0.0 (2025-01)
- Initial release
- Support for 12+ PII types
- TinyLlama 1.1B base model
- LoRA fine-tuning
- Interactive testing mode
- Production-ready Python integration
- REST API example
- Comprehensive documentation
