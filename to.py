from __future__ import annotations
from typing import Union, Any, Sequence
import polars as pl
import pandas as pd


class ColumnComparator:
    """
    Utility class for comparing columns in Polars and Pandas DataFrames.
    Returns truth series that can be used for filtering, grouping, or any other operation.
    """

    @staticmethod
    def gt(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 > column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        if isinstance(column1, pl.Series):
            return column1 > ColumnComparator._convert_to_compatible(column2, is_polars=True)
        else:  # pandas Series
            return column1 > ColumnComparator._convert_to_compatible(column2, is_polars=False)

    @staticmethod
    def lt(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 < column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        if isinstance(column1, pl.Series):
            return column1 < ColumnComparator._convert_to_compatible(column2, is_polars=True)
        else:  # pandas Series
            return column1 < ColumnComparator._convert_to_compatible(column2, is_polars=False)

    @staticmethod
    def gte(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 >= column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        if isinstance(column1, pl.Series):
            return column1 >= ColumnComparator._convert_to_compatible(column2, is_polars=True)
        else:  # pandas Series
            return column1 >= ColumnComparator._convert_to_compatible(column2, is_polars=False)

    @staticmethod
    def lte(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 <= column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        if isinstance(column1, pl.Series):
            return column1 <= ColumnComparator._convert_to_compatible(column2, is_polars=True)
        else:  # pandas Series
            return column1 <= ColumnComparator._convert_to_compatible(column2, is_polars=False)

    @staticmethod
    def eq(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 == column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        if isinstance(column1, pl.Series):
            return column1 == ColumnComparator._convert_to_compatible(column2, is_polars=True)
        else:  # pandas Series
            return column1 == ColumnComparator._convert_to_compatible(column2, is_polars=False)

    @staticmethod
    def ne(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 != column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        if isinstance(column1, pl.Series):
            return column1 != ColumnComparator._convert_to_compatible(column2, is_polars=True)
        else:  # pandas Series
            return column1 != ColumnComparator._convert_to_compatible(column2, is_polars=False)

    @staticmethod
    def contains(column: Union[pl.Series, pd.Series], pattern: str) -> Union[pl.Series, pd.Series]:
        """Return a truth series checking if a column contains a pattern."""
        ColumnComparator._validate_first_arg_is_series(column)
        if isinstance(column, pl.Series):
            return column.str.contains(pattern)
        else:  # pandas Series
            return column.str.contains(pattern)

    @staticmethod
    def not_contains(column: Union[pl.Series, pd.Series], pattern: str) -> Union[pl.Series, pd.Series]:
        """Return a truth series checking if a column does not contain a pattern."""
        ColumnComparator._validate_first_arg_is_series(column)
        if isinstance(column, pl.Series):
            return ~column.str.contains(pattern)
        else:  # pandas Series
            return ~column.str.contains(pattern)

    @staticmethod
    def is_in(column: Union[pl.Series, pd.Series], values: Sequence[Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series checking if column values are in a given sequence."""
        ColumnComparator._validate_first_arg_is_series(column)
        if isinstance(column, pl.Series):
            return column.is_in(values)
        else:  # pandas Series
            return column.isin(values)

    @staticmethod
    def not_in(column: Union[pl.Series, pd.Series], values: Sequence[Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series checking if column values are NOT in a given sequence."""
        ColumnComparator._validate_first_arg_is_series(column)
        if isinstance(column, pl.Series):
            return ~column.is_in(values)
        else:  # pandas Series
            return ~column.isin(values)

    @staticmethod
    def _validate_first_arg_is_series(column):
        """Validate that the first argument is specifically a Series (Polars or Pandas)."""
        if not isinstance(column, (pl.Series, pd.Series)):
            raise ValueError("First argument must be a Polars Series or Pandas Series")

    @staticmethod
    def _convert_to_compatible(value, is_polars: bool):
        """Convert value to a format compatible with the target Series type."""
        # If the value is already a Series of the right type, return it
        if is_polars and isinstance(value, pl.Series):
            return value
        if not is_polars and isinstance(value, pd.Series):
            return value
            
        # Handle conversion between types
        if is_polars and isinstance(value, pd.Series):
            # Convert pandas Series to compatible format for polars comparison
            return value.values
        elif not is_polars and isinstance(value, pl.Series):
            # Convert polars Series to compatible format for pandas comparison
            return value.to_pandas()
        
        # For scalar values or other types, no conversion needed
        return value


# Example usage
def example():
    # Example with pandas DataFrame
    pdf = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": ["apple", "banana", "cherry", "date", "elderberry"]
    })

    # Example with polars DataFrame
    pldf = pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": ["apple", "banana", "cherry", "date", "elderberry"]
    })

    # Get truth series for filtering in pandas
    print("\nPandas: Filtering with truth series:")
    mask = ColumnComparator.gt(pdf["A"], 3)
    filtered_pdf = pdf[mask]
    print(filtered_pdf)

    # Get truth series for filtering in polars
    print("\nPolars: Filtering with truth series:")
    mask = ColumnComparator.gt(pldf["A"], pldf["B"]/10)
    filtered_pldf = pldf.filter(mask)
    print(filtered_pldf)

    # Use with pandas groupby
    print("\nPandas: Using truth series with groupby:")
    mask = ColumnComparator.contains(pdf["C"], "a")
    group_counts = pdf.groupby(mask)["A"].sum()
    print(group_counts)

    # Use with polars groupby
    print("\nPolars: Using truth series with groupby:")
    mask = ColumnComparator.contains(pldf["C"], "a")
    group_counts = pldf.group_by(mask).agg(pl.sum("A"))
    print(group_counts)

    # This should raise an error (scalar as first argument)
    try:
        print("\nThis should fail - scalar as first argument:")
        ColumnComparator.gte(40, pdf["B"])
    except ValueError as e:
        print(f"Correctly caught error: {e}")


if __name__ == "__main__":
    example()