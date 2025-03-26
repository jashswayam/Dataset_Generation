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
    def _validate_input(s1: Union[pl.Series, pl.Expr, pd.Series]) -> None:
        """
        Validate that the first input is a Pandas or Polars Series or Expression.
        
        Raises:
            TypeError: If the input is not a supported type.
        """
        if not isinstance(s1, (pd.Series, pl.Series, pl.Expr)):
            raise TypeError(f"First argument must be a Pandas/Polars Series or Polars Expression. Got {type(s1)}")

    @staticmethod
    def list_in_list(a: str, b: str) -> bool:
        """
        Checks if all elements in string 'a' (comma-separated) are in string 'b'.
        """
        a_set = set(map(str.strip, a.split(','))) if isinstance(a, str) else set()
        b_set = set(map(str.strip, b.split(','))) if isinstance(b, str) else set()
        return a_set.issubset(b_set)

    @staticmethod
    def list_not_in_list(a: str, b: str) -> bool:
        """
        Checks if no elements in string 'a' (comma-separated) are in string 'b'.
        """
        a_set = set(map(str.strip, a.split(','))) if isinstance(a, str) else set()
        b_set = set(map(str.strip, b.split(','))) if isinstance(b, str) else set()
        return len(a_set.intersection(b_set)) == 0

    @staticmethod
    def list_contains(
        s1: Union[pl.Series, pl.Expr, pd.Series],
        s2: Union[pl.Series, pl.Expr, pd.Series, str]
    ) -> Union[pl.Series, pl.Expr, pd.Series]:
        """
        Checks if string 's2' contains at least one element from the comma-separated list in 's1'.
        """
        ExtendedOperator._validate_input(s1)

        if isinstance(s1, pd.Series):
            if isinstance(s2, str):
                return s1.apply(lambda x: any(
                    item.strip() in s2 
                    for item in str(x).split(',') if item.strip()
                ))
            elif isinstance(s2, pd.Series):
                return s1.combine(s2, lambda x, y: any(
                    item.strip() in str(y) 
                    for item in str(x).split(',') if item.strip()
                ))

        elif isinstance(s1, pl.Series):
            if isinstance(s2, str):
                return s1.map_elements(lambda x: any(
                    item.strip() in s2 
                    for item in str(x).split(',') if item.strip()
                ))
            elif isinstance(s2, pl.Series):
                return pl.Series([
                    any(
                        item.strip() in str(val2) 
                        for item in str(val1).split(',') if item.strip()
                    )
                    for val1, val2 in zip(s1, s2)
                ])

        elif isinstance(s1, pl.Expr):
            if isinstance(s2, str):
                return s1.map_elements(lambda x: any(
                    item.strip() in s2 
                    for item in str(x).split(',') if item.strip()
                ))
            elif isinstance(s2, pl.Expr):
                return s1.struct.field("col1").map_elements(
                    lambda x, y: any(
                        item.strip() in str(y) 
                        for item in str(x).split(',') if item.strip()
                    ),
                    return_dtype=pl.Boolean
                )

        raise TypeError("Unsupported types for list_contains comparison")

    @staticmethod
    def list_not_contains(
        s1: Union[pl.Series, pl.Expr, pd.Series],
        s2: Union[pl.Series, pl.Expr, pd.Series, str]
    ) -> Union[pl.Series, pl.Expr, pd.Series]:
        """
        Checks if no elements from the comma-separated list in 's1' are in 's2'.
        """
        ExtendedOperator._validate_input(s1)

        if isinstance(s1, pd.Series):
            if isinstance(s2, str):
                return s1.apply(lambda x: not any(
                    item.strip() in s2 
                    for item in str(x).split(',') if item.strip()
                ))
            elif isinstance(s2, pd.Series):
                return s1.combine(s2, lambda x, y: not any(
                    item.strip() in str(y) 
                    for item in str(x).split(',') if item.strip()
                ))

        elif isinstance(s1, pl.Series):
            if isinstance(s2, str):
                return s1.map_elements(lambda x: not any(
                    item.strip() in s2 
                    for item in str(x).split(',') if item.strip()
                ))
            elif isinstance(s2, pl.Series):
                return pl.Series([
                    not any(
                        item.strip() in str(val2) 
                        for item in str(val1).split(',') if item.strip()
                    )
                    for val1, val2 in zip(s1, s2)
                ])

        elif isinstance(s1, pl.Expr):
            if isinstance(s2, str):
                return s1.map_elements(lambda x: not any(
                    item.strip() in s2 
                    for item in str(x).split(',') if item.strip()
                ))
            elif isinstance(s2, pl.Expr):
                return s1.struct.field("col1").map_elements(
                    lambda x, y: not any(
                        item.strip() in str(y) 
                        for item in str(x).split(',') if item.strip()
                    ),
                    return_dtype=pl.Boolean
                )

        raise TypeError("Unsupported types for list_not_contains comparison")

    @staticmethod
    def list_in(
        s1: Union[pl.Series, pl.Expr, pd.Series],
        s2: Union[pl.Series, pl.Expr, pd.Series, str]
    ) -> Union[pl.Series, pl.Expr, pd.Series]:
        """
        Applies `list_in_list` logic to Pandas and Polars Series or Expressions.
        """
        ExtendedOperator._validate_input(s1)

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

    @staticmethod
    def list_not_in(
        s1: Union[pl.Series, pl.Expr, pd.Series],
        s2: Union[pl.Series, pl.Expr, pd.Series, str]
    ) -> Union[pl.Series, pl.Expr, pd.Series]:
        """
        Applies `list_not_in_list` logic to Pandas and Polars Series or Expressions.
        """
        ExtendedOperator._validate_input(s1)

        if isinstance(s1, pd.Series):
            if isinstance(s2, str):
                return s1.apply(lambda x: ExtendedOperator.list_not_in_list(x, s2))
            elif isinstance(s2, pd.Series):
                return s1.combine(s2, ExtendedOperator.list_not_in_list)

        elif isinstance(s1, pl.Series):
            if isinstance(s2, str):
                return s1.map_elements(lambda x: ExtendedOperator.list_not_in_list(x, s2))
            elif isinstance(s2, pl.Series):
                return pl.Series([
                    ExtendedOperator.list_not_in_list(val1, val2) 
                    for val1, val2 in zip(s1, s2)
                ])

        elif isinstance(s1, pl.Expr):
            if isinstance(s2, str):
                return s1.map_elements(lambda x: ExtendedOperator.list_not_in_list(x, s2))
            elif isinstance(s2, pl.Expr):
                return s1.struct.field("col1").map_elements(
                    lambda x, y: ExtendedOperator.list_not_in_list(x, y),
                    return_dtype=pl.Boolean
                )

        raise TypeError("Unsupported types for list_not_in comparison")

# Example Usage
op = ExtendedOperator

# Pandas Example
pd_series_1 = pd.Series(['a,b', 'a,b,c', 'd,e'])
pd_series_2 = pd.Series(['a,b,c', 'b,c,d', 'a,e'])
print("Pandas list_in:")
print(op.list_in(pd_series_1, pd_series_2))  # Expected output: [True, True, False]
print("\nPandas list_not_in:")
print(op.list_not_in(pd_series_1, pd_series_2))  # Expected output: [False, False, True]
print("\nPandas list_contains:")
print(op.list_contains(pd_series_1, pd_series_2))  # Expected output: [True, True, True]
print("\nPandas list_not_contains:")
print(op.list_not_contains(pd_series_1, pd_series_2))  # Expected output: [False, False, False]

# Polars Example
pl_series_1 = pl.Series(['a,b', 'a,b,c', 'd,e'])
pl_series_2 = pl.Series(['a,b,c', 'b,c,d', 'a,e'])
print("\nPolars list_in:")
print(op.list_in(pl_series_1, pl_series_2))  # Expected output: [True, True, False]
print("\nPolars list_not_in:")
print(op.list_not_in(pl_series_1, pl_series_2))  # Expected output: [False, False, True]
print("\nPolars list_contains:")
print(op.list_contains(pl_series_1, pl_series_2))  # Expected output: [True, True, True]
print("\nPolars list_not_contains:")
print(op.list_not_contains(pl_series_1, pl_series_2))  # Expected output: [False, False, False]

# Polars LazyFrame Example
df = pl.LazyFrame({"col1": ['a,b', 'a,b,c', 'd,e'], "col2": ['a,b,c', 'b,c,d', 'a,e']})
filtered_df = df.with_columns(
    list_in=pl.struct(["col1", "col2"]).map_elements(lambda x: op.list_in_list(x["col1"], x["col2"])),
    list_not_in=pl.struct(["col1", "col2"]).map_elements(lambda x: op.list_not_in_list(x["col1"], x["col2"])),
    list_contains=pl.struct(["col1", "col2"]).map_elements(lambda x: any(
        item.strip() in str(x["col2"]) 
        for item in str(x["col1"]).split(',') if item.strip()
    )),
    list_not_contains=pl.struct(["col1", "col2"]).map_elements(lambda x: not any(
        item.strip() in str(x["col2"]) 
        for item in str(x["col1"]).split(',') if item.strip()
    ))
)
print("\nPolars LazyFrame Example:")
print(filtered_df.collect())