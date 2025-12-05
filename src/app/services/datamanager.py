from typing import Literal

import kagglehub
import pandas as pd

from app.services.preprocessing import Preprocessor


class DataManager:
    def __init__(self):
        self.df: pd.DataFrame | None = None

    async def load_data(self) -> pd.DataFrame:
        try:
            file_path = kagglehub.dataset_download("advaithsrao/enron-fraud-email-dataset")
            dataframe = pd.read_csv(f"{file_path}/enron_data_fraud_labeled.csv")
        except Exception as e:
            raise e

        self.df = dataframe
        return dataframe

    async def get_basic_info(self) -> dict:
        if self.df is None:
            return {}
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "dtypes": dict(self.df.dtypes),
            "missing_values": dict(self.df.isnull().sum()),
            "duplicate_rows": self.df.duplicated().sum()
        }

    async def get_target_distribution(self) -> dict:
        if self.df is None:
            return {}

        label_column = "Label"

        if label_column not in self.df.columns:
            return {"error": "Label column not found in dataset"}

        value_counts = self.df[label_column].value_counts()

        return {
            "target_column": label_column,
            "distribution": dict(value_counts),
            "percentages": dict(value_counts / len(self.df) * 100),
            "class_balance_ratio": value_counts.min() / value_counts.max() if len(value_counts) > 1 else 1.0
        }

    async def get_text_statistics(self) -> dict:
        """Analyze text-based columns (Subject and Body)"""
        if self.df is None:
            return {}

        # Known text columns for email data
        text_columns = ["Subject", "Body"]
        text_stats = {}

        for col in text_columns:
            if col not in self.df.columns:
                text_stats[col] = {"error": f"Column '{col}' not found in dataset"}
                continue

            text_data = self.df[col].dropna().astype(str)
            if len(text_data) == 0:
                text_stats[col] = {"error": f"No valid data in column '{col}'"}
                continue

            lengths = text_data.str.len()
            word_counts = text_data.str.split().str.len()

            basic_stats = {
                "total_samples": len(text_data),
                "avg_char_length": lengths.mean(),
                "median_char_length": lengths.median(),
                "max_char_length": lengths.max(),
                "min_char_length": lengths.min(),
                "std_char_length": lengths.std(),
                "avg_word_count": word_counts.mean(),
                "median_word_count": word_counts.median(),
                "max_word_count": word_counts.max(),
                "min_word_count": word_counts.min(),
                "empty_values": (text_data == "").sum()
            }

            email_specific = {}

            if col == "Subject":
                email_specific.update({
                    "subjects_with_re": text_data.str.contains(r"^Re:", case=False, na=False).sum(),
                    "subjects_with_fwd": text_data.str.contains(r"^Fwd:", case=False, na=False).sum(),
                    "subjects_all_caps": text_data.str.isupper().sum(),
                    "subjects_with_urgent_words": text_data.str.lower().str.extract(
                        r"\b(urgent|important|asap|immediate|critical)\b"
                    ).count().iloc[0],
                })

            elif col == "Body":
                email_specific.update({
                    "contains_urls": text_data.str.contains(
                        r"https?://|www\.", case=False, na=False
                    ).sum(),
                    "contains_email_addresses": text_data.str.contains(
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", na=False
                    ).sum(),
                    "contains_phone_numbers": text_data.str.contains(
                        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", na=False
                    ).sum(),
                    "contains_money_symbols": text_data.str.contains(
                        r"[\$£€¥]|\bmoney\b|\bcash\b|\bpayment\b", case=False, na=False
                    ).sum(),
                    "avg_sentences": text_data.str.split(r"[.!?]+").str.len().mean(),
                    "contains_attachments_mention": text_data.str.contains(
                        r"\battach|\bdocument\b|\bfile\b|\bdownload\b", case=False, na=False
                    ).sum()
                })

            text_stats[col] = {**basic_stats, **email_specific}

        return text_stats

    async def get_email_specific_stats(self) -> dict:
        if self.df is None:
            return {}

        stats = {}

        email_cols = {
            "sender": "X-From",
            "recipient": "X-To",
            "subject": "Subject",
            "body": "Body"
        }

        for category, col_name in email_cols.items():
            if col_name in self.df.columns:
                unique_count = self.df[col_name].nunique()
                null_count = self.df[col_name].isnull().sum()
                total_count = len(self.df)

                base_stats = {
                    "column_name": col_name,
                    "unique_values": unique_count,
                    "null_values": null_count,
                    "null_percentage": null_count / total_count * 100,
                    "total_samples": total_count
                }

                # Add specific analysis for sender/recipient columns
                if category in ["sender", "recipient"]:
                    non_null_data = self.df[col_name].dropna()
                    if len(non_null_data) > 0:
                        # Email domain analysis
                        email_domains = non_null_data.str.extract(r"@([^@\s]+)", expand=False).dropna()
                        domain_counts = email_domains.value_counts()

                        base_stats.update({
                            "top_domains": dict(domain_counts.head(10)),
                            "unique_domains": len(domain_counts),
                            "most_common_domain": domain_counts.index[0] if len(domain_counts) > 0 else None,
                            "most_common_domain_count": domain_counts.iloc[0] if len(domain_counts) > 0 else 0,
                            "single_use_addresses": (non_null_data.value_counts() == 1).sum(),
                            "avg_emails_per_address": len(non_null_data) / unique_count if unique_count > 0 else 0
                        })

                stats[f"{category}_stats"] = base_stats
            else:
                stats[f"{category}_stats"] = {"error": f"Column '{col_name}' not found in dataset"}

        return stats

    async def get_data_quality_report(self) -> dict:
        if self.df is None:
            return {}

        # Calculate completeness for each column
        completeness = {}
        for col in self.df.columns:
            non_null_count = self.df[col].count()
            completeness[col] = {
                "completeness_rate": non_null_count / len(self.df) * 100,
                "missing_count": len(self.df) - non_null_count
            }

        # Identify potential issues
        issues = []

        # Check for columns with high missing data
        high_missing = [col for col, stats in completeness.items() if stats["completeness_rate"] < 50]
        if high_missing:
            issues.append(f"High missing data (>50%): {', '.join(high_missing)}")

        # Check for duplicate rows
        if self.df.duplicated().sum() > 0:
            issues.append(f"Found {self.df.duplicated().sum()} duplicate rows")

        # Check for constant columns
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"Constant columns (no variance): {', '.join(constant_cols)}")

        return {
            "completeness": completeness,
            "constant_columns": constant_cols,
            "potential_issues": issues,
            "overall_quality_score":
                sum(stats["completeness_rate"] for stats in completeness.values()) / len(completeness),

        }

    async def get_comprehensive_stats(self) -> dict:
        return {
            "basic_info": await self.get_basic_info(),
            "target_distribution": await self.get_target_distribution(),
            "text_statistics": await self.get_text_statistics(),
            "email_specific_stats": await self.get_email_specific_stats(),
            "data_quality": await self.get_data_quality_report()
        }

    async def handle_quality_issues(self, drop_constants: bool = True, threshold: float = 50.0):
        """Runs preprocessing steps from preprocessing.py"""
        pre = Preprocessor(self)
        try:
            if drop_constants:
                await pre.drop_constant_columns()
            await pre.handle_missing_values(threshold=threshold)
            return self.df
        except Exception:
            raise

    async def run_feature_engineering(self, top_k_domains: int):
        """Create ML-ready feature columns (text-derived and encoded domains)."""
        pre = Preprocessor(self)
        try:
            feature_cols = await pre.create_text_features(top_k_domains=top_k_domains)
            return feature_cols
        except Exception:
            raise

    async def run_vectorization(
            self,
            text_columns: list | None = None,
            vectorizer_type: Literal["tfidf"] = "tfidf",
            ngram_range: tuple = (1, 2),
            max_features: int = 10000
        ):
        """Run text vectorization using Preprocessor and store results on self."""
        pre = Preprocessor(self)
        try:
            X = await pre.vectorize_text(
                text_columns, vectorizer_type, ngram_range, max_features
            )
            return X
        except Exception:
            raise


datamanager = DataManager()
