"""
This module provides a class to provide hints for the annotation of a SAM file.

Author: Ivan Khrop
Date: 04.01.2025
"""

# class to provide hints
class HintsProvider:
    """
    Class that contains global Flag to provide hints for LLM if required.

    Class Attribute
    ----------
    use_hint: bool
        Flag that shows if hints must be provided.
    """
    use_hint: bool = False