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
        Checks if all elements of column1 exist in column2.
        - If column2 is a string → All elements of column1 should exist in this string.
        - If column2 is a Series → Each row of column1 is checked against the corresponding row in column2.
        """
        def to_set(x):
            """Converts a comma-separated string to a set of values."""
            return set(item.strip() for item in x.split(','))

        # Case 2: column2 is a single string → Compare all rows of column1 against this set
        if isinstance(column2, str):
            column2_set = to_set(column2)

            if isinstance(column1, pl.Expr):
                return column1.map_elements(lambda x: to_set(x).issubset(column2_set))
            elif isinstance(column1, pl.Series):
                return column1.apply(lambda x: to_set(x).issubset(column2_set))
            elif isinstance(column1, pd.Series):
                return column1.apply(lambda x: to_set(x).issubset(column2_set))

        # Case 1: column2 is a Series → Compare row-wise
        elif isinstance(column1, (pl.Expr, pl.Series)) and isinstance(column2, (pl.Expr, pl.Series)):
            return column1.map_elements(
                lambda x, y: to_set(x).issubset(to_set(y)), return_dtype=pl.Boolean
            ) if isinstance(column1, pl.Expr) else pl.Series([
                to_set(col1).issubset(to_set(col2))
                for col1, col2 in zip(column1, column2)
            ])
        elif isinstance(column1, pd.Series) and isinstance(column2, pd.Series):
            return pd.Series([
                to_set(col1).issubset(to_set(col2))
                for col1, col2 in zip(column1, column2)
            ])

        raise TypeError("Unsupported types for list_in comparison")

# Example Usage
op = ExtendedOperator  # Alias for convenience

### CASE 1: Series vs Series
A = pd.Series(['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'])
B = pd.Series(['AED,INR', 'EUR', 'AED,YEN'])
print(op.list_in(A, B))  # Expected: [True, True, False]

### CASE 2: Series vs String
A = pd.Series(['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'])
B = "INR,AED"
print(op.list_in(A, B))  # Expected: [True, True, False]