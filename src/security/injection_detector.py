"""Prompt injection detector with 22+ regex patterns."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NamedTuple


class _Pattern(NamedTuple):
    pattern: re.Pattern
    name: str


INJECTION_PATTERNS: list[_Pattern] = [
    _Pattern(re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE), "ignore_previous"),
    _Pattern(re.compile(r"ignore\s+prior\s+instructions", re.IGNORECASE), "ignore_prior"),
    _Pattern(re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|context)", re.IGNORECASE), "disregard_prior"),
    _Pattern(re.compile(r"forget\s+(?:everything|all|previous)", re.IGNORECASE), "forget_previous"),
    _Pattern(re.compile(r"(?:reveal|show|print|output|tell me)\s+(?:your\s+)?(?:system\s+)?prompt", re.IGNORECASE), "prompt_leak"),
    _Pattern(re.compile(r"what\s+(?:is|are|were)\s+(?:your|the)\s+(?:original\s+)?(?:instructions|system\s+prompt)", re.IGNORECASE), "prompt_leak_question"),
    _Pattern(re.compile(r"<\s*system\s*>", re.IGNORECASE), "system_prompt_inject"),
    _Pattern(re.compile(r"<<\s*SYS\s*>>", re.IGNORECASE), "llama_system_tag"),
    _Pattern(re.compile(r"\[INST\]", re.IGNORECASE), "llama_inst_tag"),
    _Pattern(re.compile(r"\[/INST\]", re.IGNORECASE), "llama_inst_close_tag"),
    _Pattern(re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|another|an?\s+)", re.IGNORECASE), "role_switch"),
    _Pattern(re.compile(r"new\s+instructions?[:,]", re.IGNORECASE), "new_instructions"),
    _Pattern(re.compile(r"(?:DAN\s+mode|do\s+anything\s+now)", re.IGNORECASE), "dan_mode"),
    _Pattern(re.compile(r"jailbreak", re.IGNORECASE), "jailbreak"),
    _Pattern(re.compile(r"override\s+(?:your\s+)?(?:safety|content|restrictions?)", re.IGNORECASE), "override_safety"),
    _Pattern(re.compile(r"ignore\s+(?:the\s+)?transcript", re.IGNORECASE), "ignore_transcript"),
    _Pattern(re.compile(r"this\s+is\s+(?:a\s+)?(?:test|simulation|role.?play).*\s+actually", re.IGNORECASE), "conversation_inject"),
    _Pattern(re.compile(r"(?:social\s+engineer|phish|manipulate)\s+(?:the\s+)?(?:AI|model|assistant)", re.IGNORECASE), "social_engineering"),
    _Pattern(re.compile(r"translate\s+(?:this\s+)?(?:to|into)\s+(?:a\s+)?different\s+(?:language|persona)", re.IGNORECASE), "translate_attack"),
    _Pattern(re.compile(r"ignore\s+(?:all\s+)?safety\s+(?:guidelines?|rules?|filters?)", re.IGNORECASE), "ignore_safety"),
    _Pattern(re.compile(r"system\s+override", re.IGNORECASE), "system_override"),
    _Pattern(re.compile(r"(?:reveal|expose|leak)\s+(?:all\s+)?(?:training\s+data|model\s+weights|hidden\s+prompt)", re.IGNORECASE), "reveal_attack"),
    # Additional patterns for robustness
    _Pattern(re.compile(r"pretend\s+(?:you\s+are|to\s+be)\s+(?:an?\s+)?(?:evil|unethical|unrestricted)", re.IGNORECASE), "pretend_evil"),
    _Pattern(re.compile(r"act\s+as\s+(?:if\s+)?(?:you\s+have\s+)?no\s+restrictions", re.IGNORECASE), "act_no_restrictions"),
]


@dataclass
class InjectionResult:
    injection_detected: bool
    matched_patterns: list[str] = field(default_factory=list)


def detect_injection(text: str) -> InjectionResult:
    matched: list[str] = []
    for pat in INJECTION_PATTERNS:
        if pat.pattern.search(text):
            matched.append(pat.name)
    return InjectionResult(injection_detected=bool(matched), matched_patterns=matched)
