"""
Utility functions for byte-based offset calculation.
Correctly handles multi-byte characters like emojis (🚀).
"""

def get_byte_offset(text, char_offset):
    """
    Convert character offset to byte offset.

    Args:
        text (str): Original text
        char_offset (int): Character-based offset

    Returns:
        int: Byte-based offset
    """
    if char_offset <= 0:
        return 0

    # Slice text by character and encode to UTF-8 to get byte length
    substring = text[:char_offset]
    return len(substring.encode('utf-8'))

def get_char_offset_from_byte(text, byte_offset):
    """
    Convert byte offset to character offset.

    Args:
        text (str): Original text
        byte_offset (int): Byte-based offset

    Returns:
        int: Character-based offset
    """
    if byte_offset <= 0:
        return 0

    # Decode UTF-8 bytes to characters
    try:
        return len(text.encode('utf-8')[:byte_offset].decode('utf-8'))
    except UnicodeDecodeError:
        # Handle invalid byte offset safely
        return len(text)

def get_utf16_code_unit_offset(text, char_offset):
    """
    Convert character offset to UTF-16 code unit offset.
    Same as Windows Notepad length display standard.

    Args:
        text (str): Original text
        char_offset (int): Character-based offset

    Returns:
        int: UTF-16 code unit-based offset
    """
    if char_offset <= 0:
        return 0

    # Slice text by character and encode to UTF-16
    substring = text[:char_offset]
    utf16_bytes = substring.encode('utf-16le')

    # Count UTF-16 code units (2 bytes = 1 code unit)
    return len(utf16_bytes) // 2

def get_char_offset_from_utf16_code_unit(text, code_unit_offset):
    """
    Convert UTF-16 code unit offset to character offset.

    Args:
        text (str): Original text
        code_unit_offset (int): UTF-16 code unit-based offset

    Returns:
        int: Character-based offset
    """
    if code_unit_offset <= 0:
        return 0

    # Convert to UTF-16 bytes
    target_bytes = code_unit_offset * 2

    # Decode UTF-16 bytes to characters
    try:
        utf16_bytes = text.encode('utf-16le')[:target_bytes]
        return len(utf16_bytes.decode('utf-16le'))
    except UnicodeDecodeError:
        # Handle invalid offset safely
        return len(text)

def convert_utf16_offset_to_char_range(text, utf16_start, utf16_end):
    """
    Convert UTF-16 code unit range to character range.

    Args:
        text (str): Original text
        utf16_start (int): UTF-16 code unit start position
        utf16_end (int): UTF-16 code unit end position

    Returns:
        tuple: (character start position, character end position)
    """
    char_start = get_char_offset_from_utf16_code_unit(text, utf16_start)
    char_end = get_char_offset_from_utf16_code_unit(text, utf16_end)
    return char_start, char_end

def correct_offset_mapping_utf16(text, offset_mapping):
    """
    Correct offset_mapping to UTF-16 code unit-based.
    Same as Windows Notepad standard.

    Args:
        text (str): Original text
        offset_mapping (list): Offset mapping in [(start_char, end_char), ...] format

    Returns:
        list: UTF-16 code unit-based offset mapping
    """
    corrected_mapping = []

    for start_char, end_char in offset_mapping:
        if start_char is None or end_char is None:
            corrected_mapping.append((start_char, end_char))
            continue

        start_code_unit = get_utf16_code_unit_offset(text, start_char)
        end_code_unit = get_utf16_code_unit_offset(text, end_char)

        corrected_mapping.append((start_code_unit, end_code_unit))

    return corrected_mapping

def adjust_span_offsets_utf16(text, spans):
    """
    Adjust span offsets to UTF-16 code unit-based.
    Same as Windows Notepad standard.

    Args:
        text (str): Original text
        spans (list): List of span dictionaries

    Returns:
        list: Spans adjusted with UTF-16 code unit-based offsets
    """
    adjusted_spans = []

    for span in spans:
        adjusted_span = span.copy()

        if 'start_offset' in span:
            adjusted_span['start_offset'] = get_utf16_code_unit_offset(text, span['start_offset'])

        if 'end_offset' in span:
            adjusted_span['end_offset'] = get_utf16_code_unit_offset(text, span['end_offset'])

        adjusted_spans.append(adjusted_span)

    return adjusted_spans

def get_emoji_byte_size(text):
    """
    Return byte sizes of emojis in the text.

    Args:
        text (str): Text to analyze

    Returns:
        dict: Dictionary in {emoji: byte_size} format
    """
    emoji_sizes = {}

    # Simple emoji detection (based on Unicode ranges)
    for char in text:
        if is_emoji(char):
            emoji_sizes[char] = len(char.encode('utf-8'))

    return emoji_sizes

def is_emoji(char):
    """
    Check if character is an emoji.

    Args:
        char (str): Character to check

    Returns:
        bool: Whether it's an emoji
    """
    # Unicode emoji ranges
    emoji_ranges = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
        (0x1F680, 0x1F6FF),  # Transport and Map
        (0x1F1E0, 0x1F1FF),  # Regional indicator symbols
        (0x2600, 0x26FF),    # Miscellaneous symbols
        (0x2700, 0x27BF),    # Dingbats
        (0xFE00, 0xFE0F),    # Variation Selectors
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        (0x1F018, 0x1F270),  # Various other emoji ranges
    ]

    code_point = ord(char)
    for start, end in emoji_ranges:
        if start <= code_point <= end:
            return True

    return False

def correct_offset_mapping(text, offset_mapping):
    """
    Correct offset_mapping to byte-based.

    Args:
        text (str): Original text
        offset_mapping (list): Offset mapping in [(start_char, end_char), ...] format

    Returns:
        list: Byte-based offset mapping
    """
    corrected_mapping = []

    for start_char, end_char in offset_mapping:
        if start_char is None or end_char is None:
            corrected_mapping.append((start_char, end_char))
            continue

        start_byte = get_byte_offset(text, start_char)
        end_byte = get_byte_offset(text, end_char)

        corrected_mapping.append((start_byte, end_byte))

    return corrected_mapping

def adjust_span_offsets(text, spans):
    """
    Adjust span offsets to byte-based.

    Args:
        text (str): Original text
        spans (list): List of span dictionaries

    Returns:
        list: Spans adjusted with byte-based offsets
    """
    adjusted_spans = []

    for span in spans:
        adjusted_span = span.copy()

        if 'start_offset' in span:
            adjusted_span['start_offset'] = get_byte_offset(text, span['start_offset'])

        if 'end_offset' in span:
            adjusted_span['end_offset'] = get_byte_offset(text, span['end_offset'])

        adjusted_spans.append(adjusted_span)

    return adjusted_spans

# Test functions
def test_emoji_offset():
    """Test emoji offset calculation"""
    test_text = "Hello 🚀 World 🌍"

    print(f"Original text: {test_text}")
    print(f"Text length (chars): {len(test_text)}")
    print(f"Text length (bytes): {len(test_text.encode('utf-8'))}")

    # Find emoji positions
    rocket_pos = test_text.find("🚀")
    earth_pos = test_text.find("🌍")

    print(f"🚀 position: {rocket_pos} (char), {get_byte_offset(test_text, rocket_pos)} (byte)")
    print(f"🌍 position: {earth_pos} (char), {get_byte_offset(test_text, earth_pos)} (byte)")

    # Emoji byte sizes
    emoji_sizes = get_emoji_byte_size(test_text)
    for emoji, size in emoji_sizes.items():
        print(f"{emoji}: {size} bytes")

def test_utf16_code_unit_offset():
    """UTF-16 code unit offset calculation test (Windows Notepad standard)"""
    test_text = "Hello 🚀 World 🌍"

    print("\n" + "=" * 60)
    print("UTF-16 Code Unit Offset Calculation Test (Windows Notepad)")
    print("=" * 60)

    print(f"Text: '{test_text}'")
    print(f"Character length: {len(test_text)}")
    print(f"UTF-16 code unit length: {len(test_text.encode('utf-16le')) // 2}")
    print()

    # UTF-16 code unit position for each character
    print("UTF-16 code unit position for each character:")
    for i, char in enumerate(test_text):
        code_unit_pos = get_utf16_code_unit_offset(test_text, i)
        char_bytes = char.encode('utf-16le')
        code_unit_count = len(char_bytes) // 2
        print(f"Char {i}: '{char}' -> code unit {code_unit_pos} (code unit count: {code_unit_count})")

    # Emoji positions
    rocket_pos = test_text.find("🚀")
    earth_pos = test_text.find("🌍")

    print(f"\n🚀 emoji:")
    print(f"  Character position: {rocket_pos}")
    print(f"  UTF-16 code unit position: {get_utf16_code_unit_offset(test_text, rocket_pos)}")
    print(f"  Displayed length in Notepad: 2")

    print(f"\n🌍 emoji:")
    print(f"  Character position: {earth_pos}")
    print(f"  UTF-16 code unit position: {get_utf16_code_unit_offset(test_text, earth_pos)}")
    print(f"  Displayed length in Notepad: 2")

def compare_offset_methods():
    """Compare various offset calculation methods"""

    print("\n" + "=" * 60)
    print("Comparison of Various Offset Calculation Methods")
    print("=" * 60)

    test_cases = [
        "Hello 🚀 World",
        "Korean text 🌍",
        "Test 🎉 emoji 🔥",
        "No emoji text"
    ]

    for text in test_cases:
        print(f"\nText: '{text}'")
        print(f"  Python len(): {len(text)} (character count)")
        print(f"  UTF-8 bytes: {len(text.encode('utf-8'))} (actual storage)")
        print(f"  UTF-16 bytes: {len(text.encode('utf-16le'))} (Windows storage)")
        print(f"  UTF-16 code units: {len(text.encode('utf-16le')) // 2} (Notepad display)")

        # Analyze emojis if present
        emojis = [char for char in text if is_emoji(char)]
        if emojis:
            print("  Emoji analysis:")
            for emoji in set(emojis):
                pos = text.find(emoji)
                utf16_pos = get_utf16_code_unit_offset(text, pos)
                print(f"    '{emoji}': char {pos} -> code unit {utf16_pos}")

if __name__ == "__main__":
    test_emoji_offset()
