"""
Synthetic Dataset Generator for ML model training
Generates 30,000 synthetic HLD samples with 38 features
"""


import os
import numpy as np
import pandas as pd
from typing import List, Dict


class SyntheticDatasetGenerator:
    """
    Generates synthetic HLD datasets with 37 features + 1 target.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(self.random_state)

        # ✅ EXACTLY 37 FEATURES
        self.feature_names = [
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

        self.target_name = "quality_score"

    def generate(self, n_samples: int = 30000) -> pd.DataFrame:

        # TEXT METRICS
        word_count = np.random.randint(500, 5000, n_samples)
        sentence_count = np.random.randint(20, 300, n_samples)
        avg_sentence_length = word_count / np.maximum(sentence_count, 1)

        # STRUCTURAL
        header_count = np.random.randint(5, 40, n_samples)
        code_block_count = np.random.randint(0, 15, n_samples)
        table_count = np.random.randint(0, 10, n_samples)
        list_count = np.random.randint(0, 25, n_samples)
        diagram_count = np.random.randint(0, 5, n_samples)

        # CONTENT
        completeness_score = np.clip(
            header_count + table_count + code_block_count + diagram_count + list_count,
            0, 100
        )

        security_mentions = np.random.randint(0, 15, n_samples)
        scalability_mentions = np.random.randint(0, 15, n_samples)
        api_mentions = np.random.randint(0, 20, n_samples)
        database_mentions = np.random.randint(0, 20, n_samples)   # ✅ added
        performance_mentions = np.random.randint(0, 10, n_samples) # ✅ added
        monitoring_mentions = np.random.randint(0, 10, n_samples)

        # COVERAGE
        duplicate_headers = np.random.randint(0, 5, n_samples)
        header_coverage = header_count / np.maximum(sentence_count, 1)
        code_coverage = code_block_count / np.maximum(header_count, 1)
        keyword_density = (
            security_mentions + scalability_mentions + api_mentions +
            database_mentions + performance_mentions + monitoring_mentions
        ) / np.maximum(word_count, 1)
        section_density = np.random.uniform(0, 1, n_samples)

        # FLAGS
        has_architecture_section = np.random.randint(0, 2, n_samples)
        has_security_section = np.random.randint(0, 2, n_samples)
        has_scalability_section = np.random.randint(0, 2, n_samples)
        has_deployment_section = np.random.randint(0, 2, n_samples)
        has_monitoring_section = np.random.randint(0, 2, n_samples)
        has_api_spec = np.random.randint(0, 2, n_samples)
        has_data_model = np.random.randint(0, 2, n_samples)

        # ARCHITECTURE
        service_count = np.random.randint(0, 25, n_samples)
        entity_count = np.random.randint(0, 40, n_samples)
        api_endpoint_count = np.random.randint(0, 100, n_samples)

        # QUALITY SCORES
        readability = np.random.uniform(10, 50, n_samples)
        documentation_quality = np.random.uniform(20, 100, n_samples)
        technical_depth = np.random.uniform(10, 100, n_samples)
        formatting_quality = np.random.uniform(20, 100, n_samples)
        examples_count = np.random.randint(0, 10, n_samples)
        consistency_score=np.random.uniform(30,100,n_samples)
        structure_quality=np.random.uniform(40,100,n_samples)

        # TARGET
        quality_score = np.clip(np.random.normal(60, 15, n_samples), 0, 100)

        # ✅ ✅ FINAL DATAFRAME WITH EXACTLY 38 COLUMNS
        data = {
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
            "database_mentions": database_mentions,       # ✅ FIXED
            "performance_mentions": performance_mentions, # ✅ FIXED
            "monitoring_mentions": monitoring_mentions,
            "duplicate_headers": duplicate_headers,
            "header_coverage": header_coverage,
            "code_coverage": code_coverage,
            "keyword_density": keyword_density,
            "section_density": section_density,
            "has_architecture_section": has_architecture_section,
            "has_security_section": has_security_section,
            "has_scalability_section": has_scalability_section,
            "has_deployment_section": has_deployment_section,
            "has_monitoring_section": has_monitoring_section,
            "has_api_spec": has_api_spec,
            "has_data_model": has_data_model,
            "service_count": service_count,
            "entity_count": entity_count,
            "api_endpoint_count": api_endpoint_count,
            "readability": readability,
            "documentation_quality": documentation_quality,
            "technical_depth": technical_depth,
            "formatting_quality": formatting_quality,
            "examples_count": examples_count,
            "consistency_score":consistency_score,
            "structure_quality":structure_quality,
            "quality_score": quality_score
        }

        df = pd.DataFrame(data)

        # VALIDATE
        self._validate_dataset(df)

        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, encoding="utf-8")

    def _validate_dataset(self, df: pd.DataFrame) -> None:
        if df.isna().sum().sum() > 0:
            raise ValueError("Dataset contains NaN values")

        # ✅ MUST BE EXACTLY 38 COLUMNS
        if df.shape[1] != 38:
            raise ValueError(f"Dataset must have 38 columns (got {df.shape[1]})")

        if df["word_count"].min() < 100:
            raise ValueError("word_count range incorrect")

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_feature_ranges(self) -> Dict[str, tuple]:
        return {name: (0, 100) for name in self.feature_names}



