
import pandas
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import logging

from claims.config import settings

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when the data file cannot be loaded."""
    pass


class DataValidationError(Exception):
    """Raised when loaded data fails validation checks."""
    pass

class ClaimsDataLoader:
    """Loads and validates the insurance claims dataset."""
    def __init__(self) -> None:
        self.source_url = settings.SOURCE_DATA_URL
        self.columns_types: dict[str, str] = {
                                            "age": "numeric",
                                            "bmi": "numeric",
                                            "children": "numeric",
                                            "charges": "numeric",
                                            "sex": "categorical",
                                            "smoker": "categorical",
                                            "region": "categorical",
                                                }

    def load_dataset(self, path: Path) -> pd.DataFrame:
        """Load CSV and return validated DataFrame."""
        df = self._read(path)
        self._validate(df)
        df = self._clean(df)
        logger.info(f"Loaded {len(df)} claims from {path}")
        return df

    def _read(self, path: Path) -> pd.DataFrame:
        """Attempt to read the CSV. Raise a clear error if it fails."""
        if not path.exists():
            raise DataLoadError(f"Data file not found: {path}\n"
                                f"Kindly download from {self.source_url}")
        try:
            df = pd.read_csv(path)
            logging.info("Insurance dataset successfully ingested.")
        except FileNotFoundError:
            logger.info("The file was not found. Please check the path.")
        except PermissionError:
            logger.info("You do not have permission to access this file.")
        except pandas.errors.ParserError:
            logger.info("There was an issue tokenizing or parsing the file structure.")
        except UnicodeDecodeError:
            logger.info("Encoding issue detected. Try setting encoding='utf-8' or 'latin-1'.")
        except Exception as e:
            raise DataLoadError(f"Failed to read CSV <{path}>: {e}") from e
        
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Check the data is what we expect.
           1. Missing required columns
           2. Too small rows
           3. Correct columns Types
           4. Empty Columns
        """

        required_columns = set(self.columns_types.keys())
        missing_columns = required_columns - set(col.lower() for col in df.columns)

        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")

        if len(df) < settings.MIN_ROW:
            raise DataValidationError(
                f"Dataset too small: {len(df)} rows." 
                f"Expected at least {settings.MIN_ROW}."
                )

        for col, col_type in self.columns_types.items():
            if col_type == "numeric":
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise DataValidationError(
                            f"Column '{col}' should be numeric but contains non-numeric values."
                        )

        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            raise DataValidationError(
                f"These columns are completely empty: {empty_cols}"
            )
    
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning - Remove whitespaces and duplicate"""
        df = df.copy()
        df = self._normalize_strings(df)
        df = self._remove_duplicates(df)
        df = self._add_claim_ids(df)
        df = df.reset_index(drop=True)
        return df

    def _normalize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize object types - remove white spaces"""
        df = df.copy()
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip().str.lower()
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            logger.warning(f"Removed {before - after} duplicate rows")
        return df
    
    def _add_claim_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a stable unique claim ID to each row."""
        df["claim_id"] = df.apply(self._generate_claim_id, axis=1)
        return df
    
    @staticmethod
    def _generate_claim_id(row: pd.Series) -> str:
        """Generate a stable claim ID from the row contents."""
        row_string = "|".join(map(str, row.values))
        digest = hashlib.sha256(row_string.encode("utf-8")).hexdigest()
        return f"CLM-{digest[:12].upper()}"

def load_data(path: Path) -> pd.DataFrame:
    """Convenience function. Wraps ClaimsDataLoader for simple usage."""
    return ClaimsDataLoader().load_dataset(path)






    