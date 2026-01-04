"""
Adversarial Anonymization Module

This module implements the feedback-guided adversarial anonymization technique
from the paper "Language Models are Advanced Anonymizers".

The technique uses LLM inference to guide anonymization, iterating through multiple
rounds where an adversarial model tries to infer personal information, and the
anonymizer uses that feedback to improve the anonymization.
"""

from typing import Optional, Dict, Any, List
from src.llm.client import call_api_with_param, set_api_provider
import time


# ============================================================
# Default Configuration - TAB Categories
# ============================================================

# TAB (Text Anonymization Benchmark) dataset-based PII categories (8 categories)
DEFAULT_TARGET_ATTRIBUTES = [
    "PERSON",      # Name, nickname, alias, username, initials
    "CODE",        # Identification numbers (SSN, phone, passport, license plate, etc.)
    "LOC",         # Place, address, geographic location
    "ORG",         # Organization, institution, company
    "DEM",         # Demographics (age, gender, occupation, education, ethnicity, physical features)
    "DATETIME",    # Date, time, duration
    "QUANTITY",    # Amount, quantity, percentage
    "MISC"         # Other identifying info (trademark, product, event, contract, law, etc.)
]


class AdversarialAnonymizer:
    """
    Implements feedback-guided adversarial anonymization.

    This anonymizer takes text and personal information inferences, then produces
    anonymized text that prevents those inferences while preserving the original
    meaning as much as possible.
    """

    def __init__(
        self,
        api_provider: str = "openai",
        anonymizer_model: str = "gpt-4-1106-preview",
        inference_model: str = "gpt-4-1106-preview",
        prompt_level: int = 3
    ):
        """
        Initialize the adversarial anonymizer.

        Args:
            api_provider: API provider ("openai", "anthropic", or "ollama")
            anonymizer_model: Model to use for anonymization
            inference_model: Model to use for inference attacks
            prompt_level: Prompt sophistication (1: naive, 2: better, 3: CoT)
        """
        # Set API provider and model using llm.client
        set_api_provider(api_provider, anonymizer_model)

        self.api_provider = api_provider
        self.anonymizer_model = anonymizer_model
        self.inference_model = inference_model
        self.prompt_level = prompt_level

    def _call_llm(self, messages: List[Dict[str, str]], model: str) -> str:
        """Call the LLM API with retry logic using llm.client"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Temporarily switch to the specified model if needed
                set_api_provider(self.api_provider, model)

                # Extract the user prompt from messages
                # Messages format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
                prompt = ""
                for msg in messages:
                    if msg["role"] == "system":
                        prompt += msg["content"] + "\n\n"
                    elif msg["role"] == "user":
                        prompt += msg["content"]

                # Call the unified API
                response = call_api_with_param(prompt)
                return response

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                raise

    def _create_inference_prompt(self, text: str, target_attributes: List[str]) -> str:
        """
        Create a prompt for the adversarial inference model.

        Based on the paper prompt from "LLM-based Anonymization".

        Args:
            text: The text to analyze
            target_attributes: List of attributes to infer (e.g., ["age", "gender", "location"])

        Returns:
            Formatted prompt for inference
        """
        # TAB category description mapping (based on TAB guidelines, single author-focused version)
        attribute_str_map = {
            "person": "the author's name (including nickname, alias, username, and initials)",
            "code": "number or code that identifies the author (SSN, phone number, passport number, license plate)",
            "loc": "place or location associated with the author (city, area, country, address, named infrastructure)",
            "org": "organization the author is affiliated with (company, school, university, healthcare institution)",
            "dem": "the author's demographic attribute (native language, ethnicity, job title, education, age)",
            "datetime": "specific date, time, or duration related to the author",
            "quantity": "meaningful quantity related to the author (percentage, monetary value)",
            "misc": "other identifying information about the author (trademark, product, event, contract, law)"
        }

        # Special instructions (based on TAB guidelines Examples section, single author-focused context)
        special_notes = {
            "person": "Include title and honorific (Mr., Dr., etc.) as it contributes to identification. Include spelling variation and initials like 'H.H.'",
            "code": "Look for the author's ID-number, report identifier, application number (e.g., 'no. 42552/98').",
            "loc": "Include number when part of entity name (e.g., 'Pilestredet 48'). Include airport, hospital, shop, road, mountain, park where the author lives or visits.",
            "org": "Include corporate designator (AS, Co., Ltd.). Include acronym and translation (e.g., 'KCK (Kurdistan Communities Union)').",
            "dem": "Note: Pronoun (he, she) should NOT be annotated. Look for: '40 years old', 'journalist', 'diagnosed with motor neurone disease'.",
            "datetime": "Do not include preposition (on, at). Example: 'Monday, October 3, 2018', 'born in 1947', '9:48 AM'.",
            "quantity": "Include currency unit in the span. Example: '$37.5 million', '375 euros', '4267 SEK'.",
            "misc": "Include speech, radio show, programming language, contract, law, and named idea. Brand is MISC when referring to a product, ORG when referring to the entity."
        }

        # Build target_attribute_str (human-readable attribute names)
        attr_strs = []
        for attr in target_attributes:
            attr_lower = attr.lower()
            if attr_lower in attribute_str_map:
                attr_strs.append(attribute_str_map[attr_lower])
            else:
                attr_strs.append(attr)
        target_attribute_str = ", ".join(attr_strs)

        # Build footer with format specification for each attribute
        footer = "First reason step-by-step for each value going over all information in the text as well as language used. Then give your top 3 guesses for each value based on your reasoning. Follow exactly the format specified below:\n\n"

        for attr in target_attributes:
            attr_lower = attr.lower()

            # Get special notes if available
            notes_str = ""
            if attr_lower in special_notes:
                notes_str = f" {special_notes[attr_lower]}"

            footer += f"Type: {attr}\n"
            footer += f"Inference: Your detailed inference\n"
            footer += f"Guess: Your top 3 guesses in a list separated by ; .{notes_str}\n"
            footer += f"Certainty: Based on your inference how certain you are from 1-5. 1 is very uncertain only based on biases and 5 has clear evidence in the text. Only give the number.\n\n"

        prompt = f"""Let us play a guessing game. The following is text written by one author. Your task is to guess the authors {target_attribute_str}?

Text:
{text}

{footer}"""
        return prompt

    def infer_attributes(self, text: str, target_attributes: List[str]) -> Dict[str, Any]:
        """
        Use the adversarial model to infer attributes from text.

        Args:
            text: Text to analyze
            target_attributes: Attributes to infer

        Returns:
            Dictionary mapping attributes to inferences
        """
        prompt = self._create_inference_prompt(text, target_attributes)

        messages = [
            {
                "role": "system",
                "content": "You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self._call_llm(messages, self.inference_model)

        # Parse the response to extract inferences
        # Based on the original implementation in reddit.py:130-239
        inferences = self._parse_inference_response(response, target_attributes)

        return inferences

    def _parse_inference_response(self, response: str, target_attributes: List[str]) -> Dict[str, Any]:
        """
        Parse the inference response from the LLM.

        Based on the original parse_answer() in reddit.py:130-239.
        Handles multi-line inferences and parses guesses as semicolon-separated lists.

        Args:
            response: Raw response from the LLM
            target_attributes: List of expected attributes

        Returns:
            Dictionary mapping attributes to their inferences
        """
        lines = response.split('\n')
        res_dict: Dict[str, Dict[str, Any]] = {}

        type_key = "temp"
        sub_key = "temp"
        res_dict[type_key] = {}

        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue

            split_line = line.split(':')

            if len(split_line[-1]) == 0:
                split_line = split_line[:-1]

            if len(split_line) == 1:
                # Continuation of previous field
                if sub_key in res_dict[type_key]:
                    if isinstance(res_dict[type_key][sub_key], list):
                        res_dict[type_key][sub_key].append(split_line[0])
                    else:
                        res_dict[type_key][sub_key] += "\n" + split_line[0]
                else:
                    res_dict[type_key][sub_key] = split_line[0]
                continue

            if len(split_line) > 2:
                split_line = [split_line[0], ':'.join(split_line[1:])]

            key, val = split_line

            if key.lower().strip() == 'type':
                # Find closest matching attribute
                type_key = self._find_closest_attribute(val.strip(), target_attributes)
                if type_key not in res_dict:
                    res_dict[type_key] = {}
            elif key.lower().strip() == 'inference':
                sub_key = 'inference'
                res_dict[type_key][sub_key] = val.strip()
            elif key.lower().strip() == 'guess':
                sub_key = 'guess'
                # Parse guesses as semicolon-separated list
                guesses = [g.strip() for g in val.split(';') if g.strip()]
                res_dict[type_key][sub_key] = guesses
            elif key.lower().strip() == 'certainty':
                sub_key = 'certainty'
                res_dict[type_key][sub_key] = val.strip()

        # Ensure all target attributes have entries
        for attr in target_attributes:
            if attr not in res_dict:
                res_dict[attr] = {
                    'inference': 'MISSING',
                    'guess': [],
                    'certainty': '1'
                }

        # Remove temp key if exists
        if 'temp' in res_dict:
            res_dict.pop('temp')

        return res_dict

    def _find_closest_attribute(self, value: str, target_attributes: List[str]) -> str:
        """
        Find the closest matching attribute from target_attributes.

        Simple matching based on lowercase comparison and substring matching.
        For more sophisticated matching, consider using fuzzy string matching libraries.

        Args:
            value: The value to match
            target_attributes: List of valid attributes

        Returns:
            The closest matching attribute or the original value
        """
        value_lower = value.lower().strip()

        # Exact match
        for attr in target_attributes:
            if attr.lower() == value_lower:
                return attr

        # Substring match
        for attr in target_attributes:
            if value_lower in attr.lower() or attr.lower() in value_lower:
                return attr

        # Return the original value if no match found
        return value

    def _create_anonymization_prompt(
        self,
        text: str,
        inferences: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Create the anonymization prompt based on the prompt level.

        Args:
            text: Original text to anonymize
            inferences: Dictionary of inferences from adversarial model

        Returns:
            List of message dictionaries for the LLM
        """
        if self.prompt_level == 1:  # Naive
            system_prompt = "You are an helpful assistant that helps users to anonymize their text."
            header = "Below I give you text from a user as well as personal inferences made about the user that can be made from the text. Your task is to anonymize the text."
            footer = "In a new line write first a single # and then return the anonymized text."

        elif self.prompt_level == 2:  # Better prompt and explanation
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you text from a user as well as personal inferences made about the user that can be made from the text. Your task is to anonymize the text to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."

        elif self.prompt_level == 3:  # CoT with examples
            system_prompt = """You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information.
Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not.
Example: 'my husband and I' -> 'I' is also valid as it only removes information."""
            header = "Below I give you text from a user as well as personal inferences made about the user that can be made from the text. Your task is to anonymize the text to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
        else:
            raise ValueError(f"Invalid prompt_level: {self.prompt_level}. Must be 1, 2, or 3.")

        # Format inferences if provided
        # Based on original implementation in llm_anonymizers.py:40-57
        inference_string = ""
        if inferences:
            for key, inf in inferences.items():
                if key == "full_answer":
                    continue
                if isinstance(inf, dict) and 'inference' in inf and 'guess' in inf:
                    inference_string += f"Type: {key}\n"
                    inference_string += f"Inference: {inf['inference']}\n"
                    # Handle guess as list (joined by semicolon) or string
                    guess = inf['guess']
                    if isinstance(guess, list):
                        guess = "; ".join(guess)
                    inference_string += f"Guess: {guess}\n\n"

        # Construct the full prompt
        if inference_string:
            intermediate = f"\n\n{text}\n\nInferences:\n\n{inference_string}"
        else:
            intermediate = f"\n\n{text}\n"

        full_prompt = f"{header}{intermediate}\n{footer}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]

        return messages

    def anonymize(
        self,
        text: str,
        inferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Anonymize the given text.

        Args:
            text: Original text to anonymize
            inferences: Optional pre-computed inferences from adversarial model

        Returns:
            Anonymized text
        """
        messages = self._create_anonymization_prompt(text, inferences)
        response = self._call_llm(messages, self.anonymizer_model)

        # Extract the anonymized text after the # marker
        if '\n#' in response:
            parts = response.split('\n#')
            if len(parts) >= 2:
                return parts[1].strip()

        # Fallback: return the whole response if no # marker found
        return response.strip()

    def adversarial_anonymize(
        self,
        text: str,
        target_attributes: List[str],
        num_rounds: int = 3,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Perform multi-round adversarial anonymization.

        This is the main method that implements the feedback-guided approach.
        In each round:
        1. The adversarial model tries to infer attributes from the text
        2. Those inferences are used to guide the anonymization
        3. The text is anonymized to prevent those specific inferences

        Args:
            text: Original text to anonymize
            target_attributes: List of attributes to protect (e.g., ["age", "gender", "location"])
            num_rounds: Number of adversarial rounds (default: 3)
            verbose: Whether to print intermediate results

        Returns:
            Dictionary containing:
                - 'original_text': The original input text
                - 'anonymized_text': The final anonymized text
                - 'rounds': List of round-by-round results with inferences
        """
        current_text = text
        rounds = []

        for round_num in range(num_rounds):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Round {round_num + 1}/{num_rounds}")
                print(f"{'='*50}")

            # Step 1: Adversarial inference
            if verbose:
                print(f"\n[Inference Phase]")
                print(f"Current text: {current_text[:200]}...")

            inferences = self.infer_attributes(current_text, target_attributes)

            if verbose:
                print(f"\nInferences:")
                for attr, inf in inferences.items():
                    print(f"  {attr}: {inf['guess']}")
                    print(f"    Reasoning: {inf['inference'][:100]}...")

            # Step 2: Anonymization using inference feedback
            if verbose:
                print(f"\n[Anonymization Phase]")

            anonymized_text = self.anonymize(current_text, inferences)

            if verbose:
                print(f"Anonymized text: {anonymized_text[:200]}...")

            # Store round results
            rounds.append({
                'round': round_num + 1,
                'input_text': current_text,
                'inferences': inferences,
                'output_text': anonymized_text
            })

            # Update current text for next round
            current_text = anonymized_text

        return {
            'original_text': text,
            'anonymized_text': current_text,
            'rounds': rounds
        }
