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
        Checks if all elements in column1 are contained in column2.
        Both columns contain comma-separated strings like "A,B,C,D".
        If column2 is a string, it is used as a reference for comparison.
        """
        def to_set(x):
            return set(item.strip() for item in x.split(','))

        # Scalar case (column2 is a single string)
        if isinstance(column2, str):
            column2_set = to_set(column2)

            if isinstance(column1, (pl.Series, pl.Expr)):
                return column1.apply(lambda x: to_set(x).issubset(column2_set))
            elif isinstance(column1, pd.Series):
                return column1.apply(lambda x: to_set(x).issubset(column2_set))

        # Series to series comparison
        else:
            if isinstance(column1, (pl.Series, pl.Expr)):
                return pl.Series([
                    to_set(col1).issubset(to_set(col2))
                    for col1, col2 in zip(column1, column2)
                ])
            elif isinstance(column1, pd.Series):
                return pd.Series([
                    to_set(col1).issubset(to_set(col2))
                    for col1, col2 in zip(column1, column2)
                ])

        raise TypeError("Unsupported types for list_in comparison")

# Example Usage
op = ExtendedOperator  # Alias for convenience

# Pandas Example
s1 = pd.Series(["A,B", "B,C", "A,D"])
s2 = "A,B,C,D"
print(op.list_in(s1, s2))  # Checks if each row in s1 is fully in s2

# Polars Example
lf = pl.LazyFrame({"col1": ["A,B", "B,C", "A,D"]})
filtered_lf = lf.filter(op.list_in(pl.col("col1"), "A,B,C,D"))
print(filtered_lf.collect())  # Filters rows where "A,B,C,D" contains all elements of "col1"