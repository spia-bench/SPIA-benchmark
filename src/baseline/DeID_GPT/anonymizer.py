"""
DeID-GPT: Zero-shot Medical Text De-Identification

Implementation based on:
Liu et al., "DeID-GPT: Zero-shot Medical Text De-Identification by GPT-4"
arXiv:2303.11032v2

This module implements the DeID-GPT framework for de-identifying medical text
using GPT-4's zero-shot in-context learning capability with HIPAA-compliant prompts.

Best performing configuration (from paper):
- Model: GPT-4 with Explicit Prompt
- Accuracy: 0.99 on i2b2/UTHealth dataset
"""

from src.llm.client import call_api_with_param, set_api_provider


# Optimal explicit prompt template from the paper (Figure 8)
# This prompt achieved 0.99 accuracy with GPT-4
DEID_GPT_EXPLICIT_PROMPT_ORIGINAL = """Please anonymize the following clinical note. Replace all the following information with the term "[redacted]":
1. Redact any strings that might be a name or acronym or initials, patients' names, doctors' names, the names of the M.D. or Dr.
2. Redact any pager names, medical staff names
3. Redact any strings that might be a location or address, such as "3970 Longview Drive"
4. Redact any strings that look like "something years old" or "age 37"
5. Redact any dates and IDs and numbers and record dates
6. Redact clinic and hospital names
7. Redact professions such as "manager"
8. Redact any contact information

Text to anonymize:
"""

# TAB (Text Anonymization Benchmark) category-based prompt (8 categories)
DEID_GPT_EXPLICIT_PROMPT = """Please anonymize the following text. Replace all the following information with the term "[redacted]":
1. Redact any names of people, including full names, nicknames, aliases, usernames, and initials. Include titles and honorifics (Mr., Dr., etc.).
2. Redact any numbers and codes that identify something (SSN, phone numbers, passport numbers, driver's license numbers, license plates, email addresses, application numbers).
3. Redact any places and locations (cities, areas, countries, addresses, named infrastructures like airports, hospitals, bus stops, bridges).
4. Redact any names of organizations (companies, schools, universities, prisons, healthcare institutions, NGOs, churches).
5. Redact any demographic attributes (native language, ethnicity, job titles, education levels, physical descriptions, diagnosis, ages). Do not redact pronouns (he, she).
6. Redact any specific dates, times, or durations. Do not include prepositions (on, at).
7. Redact any meaningful quantities (percentages, monetary values). Include currency units.
8. Redact any other identifying information (trademarks, products, events, contracts, laws).

Text to anonymize:
"""


class DeIDGPTAnonymizer:
    """
    DeID-GPT Anonymizer for medical text de-identification.

    This class implements the DeID-GPT framework as described in the paper,
    using GPT-4's zero-shot learning capability with carefully designed
    HIPAA-compliant prompts to de-identify Protected Health Information (PHI).

    The 18 HIPAA identifiers covered:
    1. Names
    2. Geographic data (addresses)
    3. Dates (except year)
    4. Phone numbers
    5. Fax numbers
    6. Email addresses
    7. Social Security numbers
    8. Medical record numbers
    9. Health plan beneficiary numbers
    10. Account numbers
    11. Certificate/license numbers
    12. Vehicle identifiers
    13. Device identifiers
    14. Web URLs
    15. IP addresses
    16. Biometric identifiers
    17. Full-face photographs
    18. Any other unique identifying number

    Attributes:
        api_provider: API provider (openai, anthropic, ollama)
        model: Model to use (default: gpt-4)
        temperature: Sampling temperature (default: 0.05 for consistency)
        max_tokens: Maximum tokens for response
    """

    def __init__(
        self,
        api_provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.05,
    ):
        self.api_provider = api_provider
        self.model = model
        self.temperature = temperature

    def anonymize(self, text: str) -> str:
        prompt = DEID_GPT_EXPLICIT_PROMPT + text

        # Set API provider before calling
        set_api_provider(self.api_provider, self.model)
        response = call_api_with_param(prompt, temperature=self.temperature)

        return response

