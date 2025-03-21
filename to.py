from __future__ import annotations
from typing import Union, Any, Sequence
import polars as pl
import pandas as pd


class ColumnComparator:
    """
    Extension methods for Polars and Pandas DataFrames to generate comparison truth series.
    Returns a truth Series of the same type as the input (pl.Series or pd.Series).
    """

    @staticmethod
    def gt(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 > column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a > b)

    @staticmethod
    def lt(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 < column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a < b)

    @staticmethod
    def gte(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 >= column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a >= b)

    @staticmethod
    def lte(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 <= column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a <= b)

    @staticmethod
    def eq(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 == column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a == b)

    @staticmethod
    def ne(column1: Union[pl.Series, pd.Series], column2: Union[pl.Series, pd.Series, Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series for column1 != column2."""
        ColumnComparator._validate_first_arg_is_series(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a != b)

    @staticmethod
    def contains(column: Union[pl.Series, pd.Series], pattern: str) -> Union[pl.Series, pd.Series]:
        """Return a truth series checking if a column contains a pattern."""
        ColumnComparator._validate_first_arg_is_series(column)
        if isinstance(column, pl.Series):
            # For Polars, use lazy evaluation
            return column.str.contains(pattern)
        else:  # pandas Series
            return column.str.contains(pattern)

    @staticmethod
    def not_contains(column: Union[pl.Series, pd.Series], pattern: str) -> Union[pl.Series, pd.Series]:
        """Return a truth series checking if a column does not contain a pattern."""
        ColumnComparator._validate_first_arg_is_series(column)
        if isinstance(column, pl.Series):
            # For Polars, use lazy evaluation
            return ~column.str.contains(pattern)
        else:  # pandas Series
            return ~column.str.contains(pattern)

    @staticmethod
    def is_in(column: Union[pl.Series, pd.Series], values: Sequence[Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series checking if column values are in a given sequence."""
        ColumnComparator._validate_first_arg_is_series(column)
        if isinstance(column, pl.Series):
            # For Polars
            return column.is_in(values)
        else:  # pandas Series
            return column.isin(values)

    @staticmethod
    def not_in(column: Union[pl.Series, pd.Series], values: Sequence[Any]) -> Union[pl.Series, pd.Series]:
        """Return a truth series checking if column values are NOT in a given sequence."""
        ColumnComparator._validate_first_arg_is_series(column)
        if isinstance(column, pl.Series):
            # For Polars
            return ~column.is_in(values)
        else:  # pandas Series
            return ~column.isin(values)

    @staticmethod
    def _validate_first_arg_is_series(column):
        """Validate that the first argument is specifically a Series (Polars or Pandas)."""
        if not isinstance(column, (pl.Series, pd.Series)):
            raise ValueError("First argument must be a Polars Series or Pandas Series")

    @staticmethod
    def _compare_columns(column1: Union[pl.Series, pd.Series], 
                         column2: Union[pl.Series, pd.Series, Any], 
                         comparison_func) -> Union[pl.Series, pd.Series]:
        """Helper method to handle different column comparison scenarios."""
        # Preserve the type of the first column (polars or pandas)
        if isinstance(column1, pl.Series):
            # Handle Polars Series comparison cases
            if isinstance(column2, pl.Series):
                return comparison_func(column1, column2)
            elif isinstance(column2, pd.Series):
                # Convert pandas Series to polars Series
                pl_series2 = pl.Series(column2.name, column2.values)
                return comparison_func(column1, pl_series2)
            else:
                # Scalar comparison
                return comparison_func(column1, column2)
        else:  # column1 is pd.Series
            # Handle Pandas Series comparison cases
            if isinstance(column2, pd.Series):
                return comparison_func(column1, column2)
            elif isinstance(column2, pl.Series):
                # Convert polars Series to pandas Series
                pd_series2 = pd.Series(column2.to_list(), name=column2.name)
                return comparison_func(column1, pd_series2)
            else:
                # Scalar comparison
                return comparison_func(column1, column2)


# Extend DataFrame classes with filter_by method
def extend_dataframes():
    """Extend Polars and Pandas DataFrames with filter_by method."""

    # For Polars DataFrame
    def polars_filter_by(self, truth_series):
        if not isinstance(truth_series, pl.Series):
            raise TypeError("Filter must be a Polars Series for Polars DataFrame")
        return self.filter(truth_series)

    # For Pandas DataFrame
    def pandas_filter_by(self, truth_series):
        if not isinstance(truth_series, pd.Series):
            raise TypeError("Filter must be a Pandas Series for Pandas DataFrame")
        return self[truth_series]

    # Add methods to the classes
    if not hasattr(pl.DataFrame, "filter_by"):
        pl.DataFrame.filter_by = polars_filter_by

    if not hasattr(pd.DataFrame, "filter_by"):
        pd.DataFrame.filter_by = pandas_filter_by


# Call this function to extend DataFrame classes
extend_dataframes()


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

    # Using Series objects directly with pandas
    print("\nPandas: Rows where A > 3:")
    truth_series = ColumnComparator.gt(pdf["A"], 3)
    filtered_pdf = pdf.filter_by(truth_series)
    print(filtered_pdf)

    # Using Series objects with polars
    print("\nPolars: Rows where A > B/10:")
    truth_series = ColumnComparator.gt(pldf["A"], pldf["B"]/10)
    filtered_pldf = pldf.filter_by(truth_series)
    print(filtered_pldf)

    # String operation example with pandas
    print("\nPandas: Rows where C contains 'a':")
    truth_series = ColumnComparator.contains(pdf["C"], "a")
    filtered_pdf = pdf.filter_by(truth_series)
    print(filtered_pdf)

    # This should raise an error (scalar as first argument)
    try:
        print("\nThis should fail - scalar as first argument:")
        ColumnComparator.gte(40, pdf["B"])
    except ValueError as e:
        print(f"Correctly caught error: {e}")


if __name__ == "__main__":
    example()