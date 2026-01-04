import os
from openai import OpenAI
from anthropic import Anthropic
from config.settings import OPENAI_API_KEY, ANTHROPIC_API_KEY, DEFAULT_API_PROVIDER, DEFAULT_MODEL_NAME

# Ollama configuration from environment variables
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
from config.prompts.analysis_subjects import ANALYSIS_AND_COUNT_SUBJECTS_TAB, ANALYSIS_AND_COUNT_SUBJECTS_PANORAMA
from config.prompts.annotate_pii import ANNOTATE_CODE_PII, ANNOTATE_NON_CODE_PII
from config.prompts.align_subjects import ALIGN_SUBJECTS_ACROSS_ANNOTATIONS, ALIGN_SUBJECTS_ACROSS_ANNOTATIONS_ANON
from config.prompts.eval_annotation_agreement import EVALUATE_ANNOTATION_AGREEMENT_GT, EVALUATE_ANNOTATION_AGREEMENT_AB
import re
import warnings

# Anthropic model max output tokens limits
ANTHROPIC_MAX_TOKENS = {
    # Claude 4.x series
    "claude-opus-4": 32000,
    "claude-opus-4.1": 32000,
    "claude-opus-4-5": 64000,
    "claude-sonnet-4": 64000,
    "claude-sonnet-4-5": 64000,
    # Claude 3.x series
    "claude-3-7-sonnet": 64000,  # 128K with extended thinking
    "claude-3-5-sonnet": 8192,
    "claude-3-5-haiku": 8192,
    "claude-3-haiku": 4096,
    "claude-3-opus": 4096,
    "claude-3-sonnet": 4096,
}

# Default max tokens for unknown Anthropic models
ANTHROPIC_DEFAULT_MAX_TOKENS = 32000


def get_anthropic_max_tokens(model_name: str) -> int:
    """Get max tokens for Anthropic model, matching by prefix (ignoring date suffix)."""
    # First try exact match
    if model_name in ANTHROPIC_MAX_TOKENS:
        return ANTHROPIC_MAX_TOKENS[model_name]

    # Try prefix match (e.g., "claude-3-5-haiku-20241022" matches "claude-3-5-haiku")
    for key in ANTHROPIC_MAX_TOKENS:
        if model_name.startswith(key):
            return ANTHROPIC_MAX_TOKENS[key]

    return ANTHROPIC_DEFAULT_MAX_TOKENS

# API Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
ollama_client = OpenAI(api_key='ollama',
                       base_url=f'{OLLAMA_BASE_URL}/v1',
                       timeout=300.0  # 5 min timeout
                       )

# Model configurations
# Supported models:
# - OpenAI: 'gpt-4.1', 'gpt-4.1-mini', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano'
# - Anthropic: 'claude-sonnet-4', 'claude-sonnet-4-5', 'claude-3-5-haiku'

# Default settings
api_provider = DEFAULT_API_PROVIDER
model_name = DEFAULT_MODEL_NAME


def set_api_provider(provider, model):
    """Set the API provider and model to use

    Args:
        provider: 'openai' or 'anthropic'
        model: Actual model name (e.g., 'gpt-4.1', 'claude-sonnet-4')
    """
    global api_provider, model_name

    valid_providers = ["openai", "anthropic", "ollama"]
    if provider not in valid_providers:
        raise ValueError(f"Invalid provider: {provider}. Must be one of {valid_providers}")

    # Auto-detect provider for Claude models (Claude only works with Anthropic API)
    if "claude" in model.lower() and provider != "anthropic":
        print(f"Warning: Model {model} is Claude model, overriding provider to 'anthropic'")
        provider = "anthropic"

    api_provider = provider
    model_name = model


def call_api(prompt):
    """Internal function to call the appropriate API based on provider"""
    if api_provider == "openai":
        completion = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    elif api_provider == "anthropic":
        message = anthropic_client.messages.create(
            model=model_name,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # Anthropic returns content as a list of content blocks
        return message.content[0].text
    elif api_provider == "ollama":
        completion = ollama_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    else:
        raise ValueError(f"Unknown API provider: {api_provider}")


# Models that don't support custom temperature (only default=1)
MODELS_WITHOUT_TEMPERATURE_SUPPORT = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
]


def call_api_with_param(prompt, **kwargs):
    """Call the appropriate API with optional parameters

    Args:
        prompt: The prompt to send to the API
        **kwargs: Additional parameters (temperature, top_p, top_k, max_tokens, etc.)

    Returns:
        str: The API response content
    """
    # Get temperature with default 0.1
    temperature = kwargs.get("temperature", 0.1)

    # Check if model supports temperature parameter
    use_temperature = model_name not in MODELS_WITHOUT_TEMPERATURE_SUPPORT

    if api_provider == "openai":
        # Extract OpenAI-supported parameters
        openai_params = {"temperature": temperature} if use_temperature else {}
        if "top_p" in kwargs:
            openai_params["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            openai_params["max_tokens"] = kwargs["max_tokens"]

        completion = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            **openai_params
        )
        return completion.choices[0].message.content

    elif api_provider == "anthropic":
        # Extract Anthropic-supported parameters
        # Note: top_k is not passed (top_k=0 causes empty responses)
        # Note: top_p is not passed (cannot be used together with temperature)
        anthropic_params = {"temperature": temperature}

        # Get max_tokens with model-specific cap
        requested_max_tokens = kwargs.get("max_tokens", 4096)
        model_max_tokens = get_anthropic_max_tokens(model_name)

        if requested_max_tokens > model_max_tokens:
            warnings.warn(
                f"Requested max_tokens ({requested_max_tokens}) exceeds {model_name} limit ({model_max_tokens}). "
                f"Capping to {model_max_tokens}."
            )
            requested_max_tokens = model_max_tokens

        message = anthropic_client.messages.create(
            model=model_name,
            max_tokens=requested_max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
            **anthropic_params
        )
        return message.content[0].text

    elif api_provider == "ollama":
        # Extract ollama-supported parameters
        # Note: Ollama OpenAI-compatible API doesn't support top_k
        ollama_params = {"temperature": temperature}
        if "top_p" in kwargs:
            ollama_params["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            ollama_params["max_tokens"] = kwargs["max_tokens"]

        completion = ollama_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            **ollama_params
        )
        return completion.choices[0].message.content

    else:
        raise ValueError(f"Unknown API provider: {api_provider}")


def fix_json_with_llm(malformed_response, error_message):
    """
    Ask LLM to fix malformed JSON response.

    Args:
        malformed_response: The original malformed JSON string
        error_message: The parsing error message

    Returns:
        tuple: (status, fixed_response)
            - status: "success" or "error"
            - fixed_response: Fixed JSON string or error message
    """
    try:
        # Create a prompt asking the LLM to fix the JSON
        fix_prompt = f"""The following response has a JSON formatting error. Please fix it to be valid JSON with proper double quotes (") for all field names and string values.

Original Response:
{malformed_response}

Error:
{error_message}

Requirements:
1. All field names MUST use double quotes (")
2. All string values MUST use double quotes (")
3. Remove any extra text that is not part of the JSON
4. Ensure all brackets are properly closed
5. Output ONLY the fixed JSON, nothing else

Fixed JSON:"""

        fixed_response = call_api_with_param(fix_prompt)

        # Clean the response
        fixed_response = fixed_response.strip()
        fixed_response = fixed_response.replace("```json", "")
        fixed_response = fixed_response.replace("```", "")
        fixed_response = fixed_response.strip()

        return "success", fixed_response

    except Exception as e:
        return "error", f"Failed to fix JSON with LLM: {str(e)}"


def get_subjects_analysis(text, dataset_type="tab"):
    """Get subjects analysis using dataset-specific prompt

    Args:
        text: The text to analyze
        dataset_type: "tab" for TAB/ECHR data, "panorama" for PANORAMA data

    Returns:
        tuple: (status, response)
    """
    try:
        if dataset_type.lower() == "panorama":
            prompt = ANALYSIS_AND_COUNT_SUBJECTS_PANORAMA.format(text=text)
        else:  # default to TAB
            prompt = ANALYSIS_AND_COUNT_SUBJECTS_TAB.format(text=text)
        response = call_api_with_param(prompt)
        return "success", response
    except Exception as e:
        return "error", f"error: {str(e)}"


def get_subject_profiling_code(text, subjects_analysis_result):
    """Get subject profiling for code-based PII (structured identifiers)."""
    try:
        # Example result_format for the prompt
        result_format = [
            {"id": 0, "description": "Subject description", "PIIs": [
                {"tag": "IDENTIFICATION_NUMBER", "keyword": "ID-123", "certainty": 5},
                {"tag": "PHONE_NUMBER", "keyword": "+1-555-0100", "certainty": 5}
            ]}
        ]
        prompt = ANNOTATE_CODE_PII.format(
            text=text,
            result_of_subjects_analysis=subjects_analysis_result,
            result_format=result_format
        )
        response = call_api_with_param(prompt)
        return "success", response
    except Exception as e:
        return "error", f"error: {str(e)}"


def get_subject_profiling_non_code(text, subjects_analysis_result):
    """Get subject profiling for non-code PII (demographic/social attributes)."""
    try:
        # Example result_format for the prompt
        result_format = [
            {"id": 0, "description": "Subject description", "PIIs": [
                {"tag": "NAME", "keyword": "John Doe", "certainty": 5},
                {"tag": "OCCUPATION", "keyword": "Engineer", "certainty": 4}
            ]}
        ]
        prompt = ANNOTATE_NON_CODE_PII.format(
            text=text,
            result_of_subjects_analysis=subjects_analysis_result,
            result_format=result_format
        )
        response = call_api_with_param(prompt)
        return "success", response
    except Exception as e:
        return "error", f"error: {str(e)}"


def get_alignment_analysis(text, annotation_a, annotation_b, anonymized_text=None):
    """
    Get subject alignment analysis across two annotations

    Args:
        text (str): The document text (original text)
        annotation_a (list): List of subjects from annotation A (GT, based on original text)
        annotation_b (list): List of subjects from annotation B (LLM, based on original or anonymized text)
        anonymized_text (str): Anonymized text. If provided, uses anon prompt template.

    Returns:
        tuple: (status, response) where status is "success" or "error"
    """
    try:
        # Format annotations for prompt
        def format_annotation(subjects):
            lines = []
            for subj in subjects:
                lines.append(f"Subject ID: {subj['id']}")
                lines.append(f"Description: {subj.get('description', '')}")
                lines.append("PIIs:")
                for pii in subj.get('PIIs', []):
                    if pii.get('keyword') and pii.get('certainty', 0) > 0:
                        lines.append(f"  - {pii['tag']}: {pii['keyword']} (certainty: {pii['certainty']})")
                lines.append("")
            return "\n".join(lines)

        annotation_a_str = format_annotation(annotation_a)
        annotation_b_str = format_annotation(annotation_b)

        # Auto-detect prompt type based on anonymized_text presence
        if anonymized_text is not None:
            prompt_template = ALIGN_SUBJECTS_ACROSS_ANNOTATIONS_ANON
            prompt = prompt_template.format(
                original_text=text,
                anonymized_text=anonymized_text,
                annotation_a=annotation_a_str,
                annotation_b=annotation_b_str
            )
        else:
            prompt_template = ALIGN_SUBJECTS_ACROSS_ANNOTATIONS
            prompt = prompt_template.format(
                text=text,
                annotation_a=annotation_a_str,
                annotation_b=annotation_b_str
            )

        response = call_api_with_param(prompt)
        return "success", response
    except Exception as e:
        return "error", f"error: {str(e)}"


def get_annotation_agreement_evaluation(keyword_a, keyword_b, evaluation_mode="inter_annotator"):
    """
    Evaluate annotation agreement between two keywords using LLM

    Args:
        keyword_a (str): First keyword (ground truth or Annotator A)
        keyword_b (str): Second keyword (prediction or Annotator B)
        evaluation_mode (str): "ground_truth" or "inter_annotator" (default)

    Returns:
        str: Raw LLM response text
    """
    # Select appropriate prompt based on evaluation mode
    if evaluation_mode == "ground_truth":
        prompt_template = EVALUATE_ANNOTATION_AGREEMENT_GT
    else:
        prompt_template = EVALUATE_ANNOTATION_AGREEMENT_AB

    try:
        prompt = prompt_template.format(
            keyword_a=keyword_a,
            keyword_b=keyword_b
        )
        response = call_api_with_param(prompt)
        return response
    except Exception as e:
        raise RuntimeError(f"LLM evaluation failed: {str(e)}")


