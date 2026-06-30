
"""
Rule-Based Quality Scorer for HLD documents
Provides rule-based quality assessment without ML
"""

import re
from dataclasses import dataclass
from typing import List, Dict


# ============================================================================
# QUALITY SCORE DATA CLASS
# ============================================================================

@dataclass
class QualityScore:
    overall_score: float
    completeness: float
    clarity: float
    consistency: float
    security: float
    recommendations: List[str]
    missing_elements: List[str]


# ============================================================================
# RULE-BASED SCORER
# ============================================================================

class RuleBasedQualityScorer:

    # ----------------------------------------------------------------------
    def __init__(self):
        """
        Initialize scoring weights and rules.
        """
        self.required_sections = [
            "architecture",
            "security",
            "scalability",
            "deployment",
            "monitoring",
            "api",
            "data model"
        ]

        # Weight contribution to final score
        self.weights = {
            "completeness": 0.35,
            "clarity": 0.25,
            "consistency": 0.20,
            "security": 0.20
        }

    # ----------------------------------------------------------------------
    def score(self, hld_markdown: str) -> QualityScore:
        """
        Main scoring method.
        Computes completeness, clarity, consistency, security, and overall score.
        """

        text = hld_markdown.lower()

        completeness_score = self.check_section_completeness(text)
        clarity_score = self.calculate_readability(text)
        consistency_score = self.check_formatting_consistency(hld_markdown)
        security_score = self.check_security_coverage(text)

        # Weighted total
        overall_score = (
            completeness_score * self.weights["completeness"] +
            clarity_score * self.weights["clarity"] +
            consistency_score * self.weights["consistency"] +
            security_score * self.weights["security"]
        )

        missing_elements = self.identify_missing_elements(text)

        recommendations = self.generate_recommendations(
            QualityScore(
                overall_score,
                completeness_score,
                clarity_score,
                consistency_score,
                security_score,
                [],
                missing_elements
            )
        )

        return QualityScore(
            overall_score=round(overall_score, 2),
            completeness=round(completeness_score, 2),
            clarity=round(clarity_score, 2),
            consistency=round(consistency_score, 2),
            security=round(security_score, 2),
            recommendations=recommendations,
            missing_elements=missing_elements
        )

    # =========================================================================
    # COMPLETENESS CHECK
    # =========================================================================
    def check_section_completeness(self, markdown: str) -> float:
        """
        Check presence of essential HLD sections.
        Returns score from 0-100.
        """
        found = 0
        for section in self.required_sections:
            if section in markdown:
                found += 1

        return (found / len(self.required_sections)) * 100

    # =========================================================================
    # READABILITY / CLARITY SCORE
    # =========================================================================
    def calculate_readability(self, text: str) -> float:
        """
        A simple readability metric based on:
        - Sentence length
        - Word length
        Score normalized to 0–100.
        """
        words = re.findall(r"\w+", text)
        sentences = re.split(r"[.!?]+", text)

        word_count = len(words)
        sentence_count = len([s for s in sentences if s.strip()])

        if word_count == 0 or sentence_count == 0:
            return 0

        avg_sentence_length = word_count / sentence_count
        avg_word_length = sum(len(w) for w in words) / len(words)

        # Convert to readability score (inverse of complexity)
        score = 100 - (avg_sentence_length * 2 + avg_word_length * 5)

        return max(0, min(100, score))

    # =========================================================================
    # FORMATTING CONSISTENCY
    # =========================================================================
    def check_formatting_consistency(self, markdown: str) -> float:
        """
        Checks formatting consistency:
        - Uniform header usage
        - Bullet list pattern consistency
        - Code block pairing
        Scored between 0-100.
        """
        score = 100

        # Header consistency
        headers = re.findall(r"^(#+)\s", markdown, flags=re.MULTILINE)
        if headers:
            lengths = [len(h) for h in headers]
            if len(set(lengths)) > 3:
                score -= 25

        # Bullet list consistency
        bullets = re.findall(r"^\s*[-*+]\s", markdown, flags=re.MULTILINE)
        if bullets and len(set([b.strip() for b in bullets])) > 1:
            score -= 25

        # Proper code blocks (paired ``` markers)
        code_blocks = len(re.findall(r"```", markdown))
        if code_blocks % 2 != 0:
            score -= 30

        return max(0, min(100, score))

    # =========================================================================
    # SECURITY COVERAGE
    # =========================================================================
    def check_security_coverage(self, text: str) -> float:
        """
        Checks if document discusses:
        - Authentication
        - Authorization
        - Encryption
        - Threat model
        """
        keywords = ["auth", "security", "jwt", "oauth", "encryption", "tls", "rbac"]

        hits = sum(text.count(k) for k in keywords)

        if hits == 0:
            return 0
        elif hits < 3:
            return 50
        else:
            return 100

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    def generate_recommendations(self, score: QualityScore) -> List[str]:
        """
        Based on scores, generate improvement suggestions.
        """
        recommendations = []

        if score.completeness < 80:
            recommendations.append("Add missing architectural or API sections.")

        if score.clarity < 70:
            recommendations.append("Improve clarity by shortening sentences and simplifying wording.")

        if score.consistency < 75:
            recommendations.append("Fix inconsistent formatting (headers, bullets, code blocks).")

        if score.security < 70:
            recommendations.append("Add details on authentication, authorization, and encryption.")

        if score.overall_score < 70:
            recommendations.append("Overall score is low—consider improving structure and depth.")

        return recommendations

    # =========================================================================
    # MISSING ELEMENT IDENTIFICATION
    # =========================================================================
    def identify_missing_elements(self, markdown: str) -> List[str]:
        """
        Identify required sections missing from the document.
        """
        missing = []
        for section in self.required_sections:
            if section not in markdown:
                missing.append(section)
        return missing

    # =========================================================================
    # WORD COUNT (HELPER)
    # =========================================================================
    def calculate_word_count(self, text: str) -> int:
        return len(re.findall(r"\w+", text))

    # =========================================================================
    # CODE COVERAGE CALCULATION
    # =========================================================================
    def calculate_code_coverage(self, markdown: str) -> float:
        """
        Number of code blocks / total headers (simple heuristic).
        """
        code_blocks = len(re.findall(r"```", markdown)) // 2
        headers = len(re.findall(r"^#+\s", markdown, flags=re.MULTILINE))

        if headers == 0:
            return 0

        return min(100, (code_blocks / headers) * 100)
