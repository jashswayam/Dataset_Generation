from __future__ import annotations
from typing import Union, Any, Sequence
import polars as pl
import pandas as pd


class ColumnComparator:
    """
    Extension methods for Polars and Pandas DataFrames to generate comparison truth series.
    Returns a Boolean Series of the same type as the input (pl.Series or pd.Series).
    """

    @staticmethod
    def gt(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 > column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        ColumnComparator._validate_compatible_types(column1, column2)
        
        # For scalar comparison or same-type series comparison
        return column1 > column2

    @staticmethod
    def lt(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 < column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        ColumnComparator._validate_compatible_types(column1, column2)
        
        return column1 < column2

    @staticmethod
    def gte(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 >= column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        ColumnComparator._validate_compatible_types(column1, column2)
        
        return column1 >= column2

    @staticmethod
    def lte(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 <= column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        ColumnComparator._validate_compatible_types(column1, column2)
        
        return column1 <= column2

    @staticmethod
    def eq(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 == column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        ColumnComparator._validate_compatible_types(column1, column2)
        
        return column1 == column2

    @staticmethod
    def ne(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 != column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        ColumnComparator._validate_compatible_types(column1, column2)
        
        return column1 != column2

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
    def _validate_compatible_types(column1, column2):
        """Validate that the columns are compatible types and don't mix polars and pandas."""
        if isinstance(column1, pl.Series) and isinstance(column2, pd.Series):
            raise TypeError("Cannot compare Polars Series with Pandas Series")
        elif isinstance(column1, pd.Series) and isinstance(column2, pl.Series):
            raise TypeError("Cannot compare Pandas Series with Polars Series")


# No need to extend DataFrames anymore since we're directly using the truth series
# that can be used with standard DataFrame filter methods in both libraries


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

    # Using with pandas - direct filter
    print("\nPandas: Rows where A > 3:")
    truth_series = ColumnComparator.gt(pdf["A"], 3)
    print(truth_series)  # Shows True/False Series
    filtered_pdf = pdf[truth_series]
    print(filtered_pdf)

    # Using with polars - direct filter
    print("\nPolars: Rows where A > B/10:")
    truth_series = ColumnComparator.gt(pldf["A"], pldf["B"]/10)
    print(truth_series)  # Shows True/False Series
    filtered_pldf = pldf.filter(truth_series)
    print(filtered_pldf)

    # String operation example with pandas
    print("\nPandas: Rows where C contains 'a':")
    truth_series = ColumnComparator.contains(pdf["C"], "a")
    filtered_pdf = pdf[truth_series]
    print(filtered_pdf)

    # Multiple conditions example with polars
    print("\nPolars: Rows where A > 2 AND C contains 'e':")
    condition1 = ColumnComparator.gt(pldf["A"], 2)
    condition2 = ColumnComparator.contains(pldf["C"], "e")
    filtered_pldf = pldf.filter(condition1 & condition2)
    print(filtered_pldf)

    # This should raise an error (scalar as first argument)
    try:
        print("\nThis should fail - scalar as first argument:")
        ColumnComparator.gte(40, pdf["B"])
    except ValueError as e:
        print(f"Correctly caught error: {e}")

    # This should raise an error (mixing pandas and polars)
    try:
        print("\nThis should fail - mixing pandas and polars:")
        ColumnComparator.eq(pdf["A"], pldf["A"])
    except TypeError as e:
        print(f"Correctly caught error: {e}")


if __name__ == "__main__":
    example()