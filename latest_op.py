import polars as pl
import pandas as pd
from typing import Union

class ExtendedOperator:
    """Custom comparison functions for Pandas and Polars."""

    @staticmethod
    def list_in(
        column1: Union[pl.Series, pl.Expr, pd.Series], 
        column2: Union[pl.Series, pl.Expr, pd.Series, str]
    ) -> Union[pl.Series, pl.Expr, pd.Series]:
        """
        Filters rows where all elements in column1 exist in column2.
        - If column2 is a string → Compare each row of column1 against this string.
        - If column2 is a Series → Compare row-wise.
        """
        def to_set(x):
            """Converts a comma-separated string to a set of values."""
            return set(x.split(',')) if isinstance(x, str) else set()

        # Case 2: column2 is a single string → Compare all rows of column1 against this set
        if isinstance(column2, str):
            column2_set = to_set(column2)

            if isinstance(column1, (pl.Series, pd.Series)):
                return column1.apply(lambda x: to_set(x).issubset(column2_set))
            elif isinstance(column1, pl.Expr):
                return column1.map_elements(lambda x: to_set(x).issubset(column2_set), return_dtype=pl.Boolean)

        # Case 1: column2 is a Series → Compare row-wise
        elif isinstance(column1, (pl.Series, pd.Series)) and isinstance(column2, (pl.Series, pd.Series)):
            return column1.apply(lambda x, y: to_set(x).issubset(to_set(y)), column2)

        elif isinstance(column1, pl.Expr) and isinstance(column2, pl.Expr):
            # FIX: Use `zip_with` instead of `map_elements`
            return column1.zip_with(column2, lambda x, y: to_set(x).issubset(to_set(y)))

        raise TypeError("Unsupported types for list_in comparison")


# Example Usage
op = ExtendedOperator()  # Alias for convenience

### 📌 Filtering in Pandas DataFrame
df_pd = pd.DataFrame({
    "A": ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
    "B": ['AED,INR', 'EUR', 'AED,YEN']
})
df_filtered_pd = df_pd[op.list_in(df_pd["A"], df_pd["B"])]
print(df_filtered_pd)  # Keeps only rows where A ⊆ B

### 📌 Filtering in Pandas DataFrame (Series vs String)
df_filtered_pd = df_pd[op.list_in(df_pd["A"], "INR,AED")]
print(df_filtered_pd)  # Keeps only rows where A ⊆ "INR,AED"

### 📌 Filtering in Polars DataFrame
df_pl = pl.DataFrame({
    "A": ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
    "B": ['AED,INR', 'EUR', 'AED,YEN']
})
df_filtered_pl = df_pl.filter(op.list_in(df_pl["A"], df_pl["B"]))
print(df_filtered_pl)  # Keeps only rows where A ⊆ B

### 📌 Filtering in Polars DataFrame (Series vs String)
df_filtered_pl = df_pl.filter(op.list_in(df_pl["A"], "INR,AED"))
print(df_filtered_pl)  # Keeps only rows where A ⊆ "INR,AED"

### 📌 Filtering in Polars LazyFrame
lf = pl.LazyFrame({
    "A": ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
    "B": ['AED,INR', 'EUR', 'AED,YEN']
})
filtered_lf = lf.filter(op.list_in(pl.col("A"), pl.col("B")))
print(filtered_lf.collect())  # Keeps only rows where A ⊆ B

### 📌 Filtering in Polars LazyFrame (Series vs String)
filtered_lf = lf.filter(op.list_in(pl.col("A"), "INR,AED"))
print(filtered_lf.collect())  # Keeps only rows where A ⊆ "INR,AED"