import polars as pl
import pandas as pd
import operator
from typing import Union

class ExtendedOperator:
    """Custom comparison functions for Pandas and Polars."""

    # Standard Python operators
    eq = operator.eq
    ne = operator.ne
    gt = operator.gt
    lt = operator.lt
    ge = operator.ge
    le = operator.le

    @staticmethod
    def list_in(
        column1: Union[pl.Series, pl.Expr, pd.Series], 
        column2: Union[pl.Series, pl.Expr, pd.Series, str]
    ) -> Union[pl.Series, pl.Expr, pd.Series]:
        """
        Checks if all elements in column2 are contained in column1.
        - Both columns contain comma-separated strings like "A,B,C,D".
        - If column2 is a string, it is used as a reference for comparison.
        - If both are Series, we perform row-wise comparison.
        """
        def to_set(x):
            """Convert a comma-separated string into a set of trimmed elements."""
            return set(item.strip() for item in x.split(',')) if isinstance(x, str) else set()

        # Case 2: If column2 is a string, compare all rows of column1 against it
        if isinstance(column2, str):
            column2_set = to_set(column2)

            if isinstance(column1, (pl.Series, pd.Series)):
                return column1.apply(lambda x: column2_set.issubset(to_set(x)))
            elif isinstance(column1, pl.Expr):
                return column1.map_elements(lambda x: column2_set.issubset(to_set(x)), return_dtype=pl.Boolean)

        # Case 1: If both are Series (row-wise comparison)
        elif isinstance(column1, pd.Series) and isinstance(column2, pd.Series):
            return column1.apply(lambda x, y: to_set(y).issubset(to_set(x)), column2)

        elif isinstance(column1, pl.Series) and isinstance(column2, pl.Series):
            return pl.Series([
                to_set(col2).issubset(to_set(col1))
                for col1, col2 in zip(column1, column2)
            ])

        elif isinstance(column1, pl.Expr) and isinstance(column2, pl.Expr):
            # ✅ Fix: Use `zip_with` for row-wise operations in LazyFrame
            return column1.zip_with(column2, lambda x, y: to_set(y).issubset(to_set(x)))

        raise TypeError("Unsupported types for list_in comparison")

# Example Usage
op = ExtendedOperator  # Alias for convenience

### 📌 Pandas Example 1: Series vs String
df_pd = pd.DataFrame({
    "A": ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR']
})
df_pd["filtered"] = op.list_in(df_pd["A"], "INR,AED")
print(df_pd)

### 📌 Pandas Example 2: Series vs Series
df_pd = pd.DataFrame({
    "A": ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
    "B": ['AED,INR', 'EUR', 'AED,YEN']
})
df_pd["filtered"] = op.list_in(df_pd["A"], df_pd["B"])
print(df_pd)

### 📌 Polars Example 1: DataFrame filtering (Series vs String)
df_pl = pl.DataFrame({
    "A": ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR']
})
df_filtered_pl = df_pl.filter(op.list_in(df_pl["A"], "INR,AED"))
print(df_filtered_pl)

### 📌 Polars Example 2: DataFrame filtering (Series vs Series)
df_pl = pl.DataFrame({
    "A": ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
    "B": ['AED,INR', 'EUR', 'AED,YEN']
})
df_filtered_pl = df_pl.filter(op.list_in(df_pl["A"], df_pl["B"]))
print(df_filtered_pl)

### 📌 Polars LazyFrame Example: Filtering (Series vs String)
lf = pl.LazyFrame({
    "A": ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR']
})
filtered_lf = lf.filter(op.list_in(pl.col("A"), "INR,AED"))
print(filtered_lf.collect())

### 📌 Polars LazyFrame Example: Filtering (Series vs Series)
lf = pl.LazyFrame({
    "A": ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
    "B": ['AED,INR', 'EUR', 'AED,YEN']
})
filtered_lf = lf.filter(op.list_in(pl.col("A"), pl.col("B")))
print(filtered_lf.collect())