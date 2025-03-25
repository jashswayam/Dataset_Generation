import polars as pl
import pandas as pd
import operator
from typing import Union

class ExtendedOperator:
    """Custom comparison functions for Pandas and Polars."""

    eq = operator.eq
    ne = operator.ne
    gt = operator.gt
    lt = operator.lt
    ge = operator.ge
    le = operator.le

    @staticmethod
    def list_in_list(a: str, b: str) -> bool:
        """
        Checks if all elements in string 'a' (comma-separated) are in string 'b'.
        """
        a_set = set(map(str.strip, a.split(','))) if isinstance(a, str) else set()
        b_set = set(map(str.strip, b.split(','))) if isinstance(b, str) else set()
        return a_set.issubset(b_set)

    @staticmethod
    def list_in(
        s1: Union[pl.Series, pl.Expr, pd.Series],
        s2: Union[pl.Series, pl.Expr, pd.Series, str]
    ) -> Union[pl.Series, pl.Expr, pd.Series]:
        """
        Applies `list_in_list` logic to Pandas and Polars Series or Expressions.
        """
        if isinstance(s1, pd.Series):
            if isinstance(s2, str):
                return s1.apply(lambda x: ExtendedOperator.list_in_list(x, s2))
            elif isinstance(s2, pd.Series):
                return s1.combine(s2, ExtendedOperator.list_in_list)
        
        elif isinstance(s1, pl.Series):
            if isinstance(s2, str):
                return s1.map_elements(lambda x: ExtendedOperator.list_in_list(x, s2))
            elif isinstance(s2, pl.Series):
                return pl.Series([
                    ExtendedOperator.list_in_list(val1, val2) 
                    for val1, val2 in zip(s1, s2)
                ])
        
        elif isinstance(s1, pl.Expr):
            if isinstance(s2, str):
                return s1.map_elements(lambda x: ExtendedOperator.list_in_list(x, s2))
            elif isinstance(s2, pl.Expr):
                return s1.struct.field("col1").map_elements(
                    lambda x, y: ExtendedOperator.list_in_list(x, y),
                    return_dtype=pl.Boolean
                )
        
        raise TypeError("Unsupported types for list_in comparison")

# Example Usage
op = ExtendedOperator

# Pandas Example
pd_series_1 = pd.Series(['a,b', 'a,b,c', 'd,e'])
pd_series_2 = pd.Series(['a,b,c', 'b,c,d', 'a,e'])
print(op.list_in(pd_series_1, pd_series_2))  # Expected output: [True, True, False]

# Polars Example
pl_series_1 = pl.Series(['a,b', 'a,b,c', 'd,e'])
pl_series_2 = pl.Series(['a,b,c', 'b,c,d', 'a,e'])
print(op.list_in(pl_series_1, pl_series_2))  # Expected output: [True, True, False]

# Polars LazyFrame Example
df = pl.LazyFrame({"col1": ['a,b', 'a,b,c', 'd,e'], "col2": ['a,b,c', 'b,c,d', 'a,e']})
filtered_df = df.with_columns(
    pl.struct(["col1", "col2"]).map_elements(lambda x: op.list_in_list(x["col1"], x["col2"]))
)
print(filtered_df.collect())
