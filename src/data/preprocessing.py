"""
Arabic text preprocessing and normalization.

This module handles cleaning and normalizing Arabic text before indexing:
- Remove diacritics (tashkeel) that don't affect meaning
- Normalize whitespace and special characters
- Remove URLs and email addresses
- Handle common Arabic text normalization issues
"""

import re
from typing import List


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritical marks (tashkeel) from text.
    
    Arabic diacritics provide pronunciation guidance but don't change word meaning.
    Removing them improves tokenization consistency across different document sources.
    
    Diacritics removed:
    - Fatha (َ) — short 'a' vowel
    - Damma (ُ) — short 'u' vowel  
    - Kasra (ِ) — short 'i' vowel
    - Sukun (ْ) — no vowel marker
    - Shadda (ّ) — doubled consonant
    - Tanwin (ً, ٌ, ٍ) — nunation/case endings
    - Hamza diacritics — various hamza placements
    
    Args:
        text (str): Arabic text with potential diacritics
    
    Returns:
        str: Text with all diacritical marks removed
    
    Example:
        >>> text = "مُحَمَّد"  # Muhammad with diacritics
        >>> remove_diacritics(text)
        'محمد'  # Muhammad without diacritics
    """
    # Arabic diacritics Unicode range: U+064B to U+0652
    diacritics_pattern = r'[\u064B-\u0652]'
    return re.sub(diacritics_pattern, '', text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Handles:
    - Multiple consecutive spaces → single space
    - Newlines and tabs → single space
    - Leading/trailing whitespace → removed
    - Non-breaking spaces (U+00A0) → regular space
    
    Args:
        text (str): Text with potentially irregular whitespace
    
    Returns:
        str: Text with normalized whitespace
    
    Example:
        >>> text = "Hello    world  \\n  test"
        >>> normalize_whitespace(text)
        'Hello world test'
    """
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\u00A0', ' ')
    
    # Replace newlines, tabs with spaces
    text = re.sub(r'[\n\t\r]', ' ', text)
    
    # Collapse multiple spaces into single space
    text = re.sub(r' +', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()


def remove_urls_and_emails(text: str) -> str:
    """
    Remove URLs and email addresses from text.
    
    Matches:
    - HTTP(S) URLs: https://example.com
    - Email addresses: user@domain.com
    - FTP URLs: ftp://example.com
    
    Args:
        text (str): Text potentially containing URLs or emails
    
    Returns:
        str: Text with URLs and emails removed
    
    Example:
        >>> text = "Visit https://example.com or email user@domain.com for info"
        >>> remove_urls_and_emails(text)
        'Visit  or email  for info'
    """
    # Remove URLs
    url_pattern = r'https?://[^\s]+'
    text = re.sub(url_pattern, '', text)
    
    # Remove email addresses
    email_pattern = r'[^\s]+@[^\s]+'
    text = re.sub(email_pattern, '', text)
    
    return text


def remove_special_characters(text: str, keep_arabic_punctuation: bool = True) -> str:
    """
    Remove special characters while preserving letters, numbers, and punctuation.
    
    Args:
        text (str): Text with potential special characters
        keep_arabic_punctuation (bool): If True, keep Arabic punctuation (؟ ، ؛ :)
    
    Returns:
        str: Text with special characters removed
    
    Notes:
        - Arabic letters: U+0600 to U+06FF
        - ASCII letters and numbers are always kept
        - Common punctuation kept: . , ! ? : ; ' " - ( ) [ ]
        - If keep_arabic_punctuation=True, also keeps ؟ (Arabic question mark)
    
    Example:
        >>> text = "Hello!!! @#$% World???"
        >>> remove_special_characters(text)
        'Hello World'
    """
    if keep_arabic_punctuation:
        # Keep Arabic/Latin letters, numbers, common punctuation, and Arabic punctuation
        pattern = r'[^\u0600-\u06FFa-zA-Z0-9\s\.\,\!\?\:\;\'\"\-\(\)\[\]؟،؛]'
    else:
        # Keep only Arabic/Latin letters, numbers, and basic punctuation
        pattern = r'[^\u0600-\u06FFa-zA-Z0-9\s\.\,\!\?\:\;\'\"\-\(\)\[\]]'
    
    return re.sub(pattern, '', text)


def normalize_arabic_characters(text: str) -> str:
    """
    Normalize variant Arabic characters to their canonical forms.
    
    Handles:
    - Alef variants (أ ، إ ، آ) → ا (regular Alef)
    - Teh Marbuta (ة) → Ha (ه) in some contexts
    - Yeh variants (ى ، يـ) → Ya (ي)
    
    This improves matching for dialectal variations and OCR errors.
    
    Args:
        text (str): Arabic text with potential character variants
    
    Returns:
        str: Text with normalized Arabic characters
    
    Example:
        >>> text = "أحمد إبراهيم آية"  # variants of alef
        >>> normalize_arabic_characters(text)
        'احمد ابراهيم اية'  # all normalized to regular alef
    """
    # Normalize Alef variants to regular Alef (ا)
    text = re.sub(r'[أإآ]', 'ا', text)
    
    # Normalize Yeh variants to regular Yeh (ي)
    # This includes Alef Maksura (ى) which is often confused with Yeh
    text = re.sub(r'[ىي]', 'ي', text)
    
    # Optional: normalize Teh Marbuta (ة) to Ha (ه)
    # Commented out because this changes meaning in some contexts
    # text = re.sub(r'ة', 'ه', text)
    
    return text


def preprocess_text(text: str, 
                   remove_diacritics_flag: bool = True,
                   normalize_chars: bool = True,
                   remove_urls: bool = True) -> str:
    """
    Complete preprocessing pipeline for Arabic text.
    
    Applies all normalization steps in sequence:
    1. Remove diacritics (if enabled)
    2. Normalize Arabic character variants (if enabled)
    3. Remove URLs and emails (if enabled)
    4. Remove special characters
    5. Normalize whitespace
    
    Args:
        text (str): Raw text to preprocess
        remove_diacritics_flag (bool): Whether to remove diacritical marks. Default: True
        normalize_chars (bool): Whether to normalize Arabic character variants. Default: True
        remove_urls (bool): Whether to remove URLs and emails. Default: True
    
    Returns:
        str: Fully preprocessed text
    
    Example:
        >>> raw = "مُحَمَّد أحمد إبراهيم يعمل في https://example.com!!! @#$%"
        >>> preprocess_text(raw)
        'محمد احمد ابراهيم يعمل في'
    """
    # Step 1: Remove diacritics
    if remove_diacritics_flag:
        text = remove_diacritics(text)
    
    # Step 2: Normalize Arabic character variants
    if normalize_chars:
        text = normalize_arabic_characters(text)
    
    # Step 3: Remove URLs and emails
    if remove_urls:
        text = remove_urls_and_emails(text)
    
    # Step 4: Remove special characters (keep Arabic punctuation)
    text = remove_special_characters(text, keep_arabic_punctuation=True)
    
    # Step 5: Normalize whitespace (final step, collapses spacing)
    text = normalize_whitespace(text)
    
    return text


def preprocess_corpus(texts: List[str], **kwargs) -> List[str]:
    """
    Apply preprocessing to a list of texts (full corpus).
    
    Useful for batch preprocessing all chunks before indexing.
    
    Args:
        texts (List[str]): List of text strings
        **kwargs: Additional arguments passed to preprocess_text()
    
    Returns:
        List[str]: List of preprocessed texts
    
    Example:
        >>> corpus = ["نص أول", "نص ثاني مع رابط https://example.com"]
        >>> preprocess_corpus(corpus)
        ['نص أول', 'نص ثاني مع رابط']
    """
    return [preprocess_text(text, **kwargs) for text in texts]