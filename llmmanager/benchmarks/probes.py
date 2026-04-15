"""Standardized quality probe prompt sets."""

from __future__ import annotations

PROBE_SETS: dict[str, list[dict[str, str]]] = {
    "coding": [
        {
            "prompt": "Write a Python function that checks if a number is prime.",
            "notes": "Tests basic code generation",
        },
        {
            "prompt": (
                "The following Python code has a bug. Find and fix it:\n\n"
                "def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n)\n"
            ),
            "notes": "Tests bug identification",
        },
        {
            "prompt": "Explain what this code does in one sentence: `[x**2 for x in range(10) if x % 2 == 0]`",
            "notes": "Tests code comprehension",
        },
    ],
    "reasoning": [
        {
            "prompt": "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops Lazzles? Explain your reasoning.",
            "notes": "Tests logical deduction",
        },
        {
            "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "notes": "Tests mathematical reasoning (classic cognitive bias test)",
        },
        {
            "prompt": "What comes next in this sequence: 2, 6, 12, 20, 30, ?",
            "notes": "Tests pattern recognition",
        },
    ],
    "instruction": [
        {
            "prompt": (
                "Do the following in order:\n"
                "1. List three fruits\n"
                "2. Reverse the list\n"
                "3. Write each item in ALL CAPS\n"
                "4. Number them starting from 10"
            ),
            "notes": "Tests multi-step instruction following",
        },
        {
            "prompt": "Respond with exactly 5 words, no more, no less: What is the capital of France?",
            "notes": "Tests constraint adherence",
        },
    ],
    "chat": [
        {
            "prompt": "I'm feeling a bit overwhelmed with work lately. Any thoughts?",
            "notes": "Tests conversational tone and empathy",
        },
        {
            "prompt": "What's the most interesting thing you know about the ocean?",
            "notes": "Tests knowledge and conversational engagement",
        },
    ],
}
