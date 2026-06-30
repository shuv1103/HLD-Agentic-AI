

"""
Feature Extractor for ML models
Extracts and processes 37 HLD features from markdown documents
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List
from pydantic import BaseModel


class HLDFeatures(BaseModel):
    """Structured container for extracted HLD features"""
    word_count: float
    sentence_count: float
    avg_sentence_length: float

    header_count: float
    code_block_count: float
    table_count: float
    list_count: float
    diagram_count: float

    completeness_score: float
    security_mentions: float
    scalability_mentions: float
    api_mentions: float
    database_mentions: float
    performance_mentions: float
    monitoring_mentions: float

    duplicate_headers: float
    header_coverage: float
    code_coverage: float
    keyword_density: float
    section_density: float

    has_architecture_section: float
    has_security_section: float
    has_scalability_section: float
    has_deployment_section: float
    has_monitoring_section: float
    has_api_spec: float
    has_data_model: float

    service_count: float
    entity_count: float
    api_endpoint_count: float

    readability: float
    documentation_quality: float
    technical_depth: float
    formatting_quality: float
    examples_count: float
    consistency_score: float
    structure_quality: float


class FeatureExtractor:
    """
    Extracts and validates 37 structured, textual, semantic, and architectural
    features from HLD markdown text.
    """

    def __init__(self):
        # ✅ 37 feature names (same as SyntheticDatasetGenerator)
        self.features = [
            "word_count",
            "sentence_count",
            "avg_sentence_length",

            "header_count",
            "code_block_count",
            "table_count",
            "list_count",
            "diagram_count",

            "completeness_score",
            "security_mentions",
            "scalability_mentions",
            "api_mentions",
            "database_mentions",
            "performance_mentions",
            "monitoring_mentions",

            "duplicate_headers",
            "header_coverage",
            "code_coverage",
            "keyword_density",
            "section_density",

            "has_architecture_section",
            "has_security_section",
            "has_scalability_section",
            "has_deployment_section",
            "has_monitoring_section",
            "has_api_spec",
            "has_data_model",

            "service_count",
            "entity_count",
            "api_endpoint_count",

            "readability",
            "documentation_quality",
            "technical_depth",
            "formatting_quality",
            "examples_count",
            "consistency_score",
            "structure_quality"
        ]

        # ✅ Default feature ranges for normalization (0–10000 wide to be safe)
        self.feature_ranges = {name: (0, 10000) for name in self.features}

    # =====================================================================
    # ✅ Primary Interface
    # =====================================================================

    def extract(self, hld_text: str) -> HLDFeatures:
        """
        Entry point for feature extraction.
        Returns a Pydantic HLDFeatures object.
        """
        if not isinstance(hld_text, str):
            raise ValueError("Invalid input: HLD text must be a string")

        if not hld_text.strip():
            hld_text = " "

        raw_features = self.extract_features(hld_text)
        return HLDFeatures(**raw_features)

    def extract_features(self, hld_text: str) -> Dict[str, float]:
        """
        Extract all 37 HLD features from markdown text.
        """

        text = self.normalize_text(hld_text)
        text_lower = text.lower()

        # TEXT METRICS
        word_count = self.count_words(text)
        sentence_count = self.count_sentences(text)
        avg_sentence_length = word_count / max(1, sentence_count)

        # STRUCTURAL
        header_count = self.count_headers(text)
        code_block_count = self.count_code_blocks(text)
        table_count = self.count_tables(text)
        list_count = len(re.findall(r"^\s*[-*+]\s", text, flags=re.MULTILINE))
        diagram_count = len(re.findall(r"```mermaid|diagram:", text_lower))

        # CONTENT & SEMANTIC
        security_mentions = self.count_mention(text_lower, ["security", "auth", "encryption"])
        scalability_mentions = self.count_mention(text_lower, ["scale", "load", "capacity"])
        api_mentions = self.count_mention(text_lower, ["api", "endpoint", "rest", "graphql"])
        database_mentions = self.count_mention(text_lower, ["database", "sql", "nosql", "db"])
        performance_mentions = self.count_mention(text_lower, ["performance", "latency", "throughput"])
        monitoring_mentions = self.count_mention(text_lower, ["monitor", "logging", "alert", "metrics"])

        # COVERAGE
        headers = re.findall(r"^#+\s+(.*)", text, flags=re.MULTILINE)
        duplicate_headers = len(headers) - len(set(headers)) if headers else 0

        completeness_score = header_count + table_count + code_block_count + diagram_count + list_count
        header_coverage = header_count / max(1, sentence_count)
        code_coverage = code_block_count / max(1, header_count)
        keyword_count = (
            security_mentions + scalability_mentions + api_mentions +
            database_mentions + performance_mentions + monitoring_mentions
        )
        keyword_density = keyword_count / max(1, word_count)

        # SECTION FLAGS
        sections = self.get_sections(text_lower)
        section_density = sum(sections.values())

        # ARCHITECTURE
        service_count = len(re.findall(r"service\s+[A-Za-z0-9_-]+", text_lower))
        entity_count = len(re.findall(r"entity\s+[A-Za-z0-9_-]+", text_lower))
        api_endpoint_count = len(re.findall(r"(GET|POST|PUT|DELETE)\s+/[A-Za-z0-9/_-]+", text))

        # QUALITY ESTIMATES (basic heuristics)
        readability = avg_sentence_length
        documentation_quality = (header_count + code_block_count) / max(1, list_count + 1)
        technical_depth = (api_mentions + database_mentions + performance_mentions) * 2
        formatting_quality = 100 - min(100, duplicate_headers * 5)
        examples_count = len(re.findall(r"example", text_lower))
        consistency_score = np.clip(100 - abs(header_count - list_count), 0, 100)
        structure_quality = np.clip((header_count + section_density * 10), 0, 100)

        features = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "header_count": header_count,
            "code_block_count": code_block_count,
            "table_count": table_count,
            "list_count": list_count,
            "diagram_count": diagram_count,
            "completeness_score": completeness_score,
            "security_mentions": security_mentions,
            "scalability_mentions": scalability_mentions,
            "api_mentions": api_mentions,
            "database_mentions": database_mentions,
            "performance_mentions": performance_mentions,
            "monitoring_mentions": monitoring_mentions,
            "duplicate_headers": duplicate_headers,
            "header_coverage": header_coverage,
            "code_coverage": code_coverage,
            "keyword_density": keyword_density,
            "section_density": section_density,
            "has_architecture_section": float(sections["architecture"]),
            "has_security_section": float(sections["security"]),
            "has_scalability_section": float(sections["scalability"]),
            "has_deployment_section": float(sections["deployment"]),
            "has_monitoring_section": float(sections["monitoring"]),
            "has_api_spec": float(sections["api"]),
            "has_data_model": float(sections["data_model"]),
            "service_count": service_count,
            "entity_count": entity_count,
            "api_endpoint_count": api_endpoint_count,
            "readability": readability,
            "documentation_quality": documentation_quality,
            "technical_depth": technical_depth,
            "formatting_quality": formatting_quality,
            "examples_count": examples_count,
            "consistency_score": consistency_score,
            "structure_quality": structure_quality
        }

        # HANDLE MISSING / SCALE
        features = self.handle_missing_values(features)
        features = self.scale_features(features)

        return features

    # =====================================================================
    # Normalization and Validation
    # =====================================================================

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text or ""
        text = text.lower()
        text = re.sub(r"[^\w\s#`|:/\-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def handle_missing_values(self, features: Dict[str, float]) -> Dict[str, float]:
        return {k: (0 if v is None or (isinstance(v, float) and np.isnan(v)) else v)
                for k, v in features.items()}

    def scale_features(self, features: Dict[str, float]) -> Dict[str, float]:
        scaled = {}
        for k, v in features.items():
            min_v, max_v = self.feature_ranges[k]
            scaled[k] = 0 if max_v == min_v else (v - min_v) / (max_v - min_v)
        return scaled

    def validate_features(self, features: Dict[str, float]) -> bool:
        for name in self.features:
            if name not in features:
                return False
            val = float(features[name])
            min_v, max_v = self.feature_ranges[name]
            if not (min_v <= val <= max_v):
                return False
        return True

    # =====================================================================
    # Helper Counters
    # =====================================================================

    @staticmethod
    def count_words(text: str) -> int:
        return len(re.findall(r"\w+", text))

    @staticmethod
    def count_sentences(text: str) -> int:
        return len([s for s in re.split(r"[.!?]+", text) if s.strip()])

    @staticmethod
    def count_headers(markdown: str) -> int:
        return len(re.findall(r"^#+\s", markdown, flags=re.MULTILINE))

    @staticmethod
    def count_code_blocks(markdown: str) -> int:
        return len(re.findall(r"```", markdown)) // 2

    @staticmethod
    def count_tables(markdown: str) -> int:
        return len(re.findall(r"\|.*\|", markdown)) // 2

    @staticmethod
    def count_mention(text: str, keywords: List[str]) -> int:
        return sum(text.count(k.lower()) for k in keywords)

    @staticmethod
    def get_sections(markdown: str) -> Dict[str, bool]:
        return {
            "architecture": "architecture" in markdown,
            "security": "security" in markdown,
            "scalability": "scalability" in markdown,
            "deployment": "deployment" in markdown,
            "monitoring": "monitoring" in markdown,
            "api": "api" in markdown or "endpoint" in markdown,
            "data_model": "data model" in markdown or "schema" in markdown
        }

    def get_feature_names(self) -> List[str]:
        return self.features

    def get_feature_ranges(self) -> Dict[str, tuple]:
        return self.feature_ranges
