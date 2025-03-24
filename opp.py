from __future__ import annotations
from typing import Union, Any, Sequence, Optional
import polars as pl
import pandas as pd


class ColumnComparator:
    """
    Extension methods for Polars and Pandas DataFrames to generate comparison truth series.
    Returns a Boolean Series of the same type as the input (pl.Series, pl.Expr, or pd.Series).
    
    Supports:
    - Polars DataFrame
    - Polars LazyFrame (using col expressions)
    - Pandas DataFrame
    """

    @staticmethod
    def gt(column1: Union[pl.Series, pl.Expr, pd.Series], column2: Union[pl.Series, pl.Expr, pd.Series, Any]) -> Union[pl.Series, pl.Expr, pd.Series]:
        """Return a truth series for column1 > column2."""
        ColumnComparator._validate_first_arg_is_valid(column1)
        ColumnComparator._validate_compatible_types(column1, column2)

        # For scalar comparison or same-type series comparison
        return column1 > column2

    @staticmethod
    def lt(column1: Union[pl.Series, pl.Expr, pd.Series], column2: Union[pl.Series, pl.Expr, pd.Series, Any]) -> Union[pl.Series, pl.Expr, pd.Series]:
        """Return a truth series for column1 < column2."""
        ColumnComparator._validate_first_arg_is_valid(column1)
        ColumnComparator._validate_compatible_types(column1, column2)

        return column1 < column2

    @staticmethod
    def gte(column1: Union[pl.Series, pl.Expr, pd.Series], column2: Union[pl.Series, pl.Expr, pd.Series, Any]) -> Union[pl.Series, pl.Expr, pd.Series]:
        """Return a truth series for column1 >= column2."""
        ColumnComparator._validate_first_arg_is_valid(column1)
        ColumnComparator._validate_compatible_types(column1, column2)

        return column1 >= column2

    @staticmethod
    def lte(column1: Union[pl.Series, pl.Expr, pd.Series], column2: Union[pl.Series, pl.Expr, pd.Series, Any]) -> Union[pl.Series, pl.Expr, pd.Series]:
        """Return a truth series for column1 <= column2."""
        ColumnComparator._validate_first_arg_is_valid(column1)
        ColumnComparator._validate_compatible_types(column1, column2)

        return column1 <= column2

    @staticmethod
    def eq(column1: Union[pl.Series, pl.Expr, pd.Series], column2: Union[pl.Series, pl.Expr, pd.Series, Any]) -> Union[pl.Series, pl.Expr, pd.Series]:
        """Return a truth series for column1 == column2."""
        ColumnComparator._validate_first_arg_is_valid(column1)
        ColumnComparator._validate_compatible_types(column1, column2)

        return column1 == column2

    @staticmethod
    def ne(column1: Union[pl.Series, pl.Expr, pd.Series], column2: Union[pl.Series, pl.Expr, pd.Series, Any]) -> Union[pl.Series, pl.Expr, pd.Series]:
        """Return a truth series for column1 != column2."""
        ColumnComparator._validate_first_arg_is_valid(column1)
        ColumnComparator._validate_compatible_types(column1, column2)

        return column1 != column2

    @staticmethod
    def contains(column: Union[pl.Series, pl.Expr, pd.Series], pattern: str) -> Union[pl.Series, pl.Expr, pd.Series]:
        """Return a truth series checking if a column contains a pattern."""
        ColumnComparator._validate_first_arg_is_valid(column)

        if isinstance(column, pl.Series):
            return column.str.contains(pattern)
        elif isinstance(column, pl.Expr):
            return column.str.contains(pattern)
        else:  # pandas Series
            return column.str.contains(pattern)

    @staticmethod
    def not_contains(column: Union[pl.Series, pl.Expr, pd.Series], pattern: str) -> Union[pl.Series, pl.Expr, pd.Series]:
        """Return a truth series checking if a column does not contain a pattern."""
        ColumnComparator._validate_first_arg_is_valid(column)

        if isinstance(column, pl.Series):
            return ~column.str.contains(pattern)
        elif isinstance(column, pl.Expr):
            return ~column.str.contains(pattern)
        else:  # pandas Series
            return ~column.str.contains(pattern)

    @staticmethod
    def is_in(column: Union[pl.Series, pl.Expr, pd.Series], values: Sequence[Any]) -> Union[pl.Series, pl.Expr, pd.Series]:
        """Return a truth series checking if column values are in a given sequence."""
        ColumnComparator._validate_first_arg_is_valid(column)

        if isinstance(column, pl.Series):
            return column.is_in(values)
        elif isinstance(column, pl.Expr):
            return column.is_in(values)
        else:  # pandas Series
            return column.isin(values)

    @staticmethod
    def not_in(column: Union[pl.Series, pl.Expr, pd.Series], values: Sequence[Any]) -> Union[pl.Series, pl.Expr, pd.Series]:
        """Return a truth series checking if column values are NOT in a given sequence."""
        ColumnComparator._validate_first_arg_is_valid(column)

        if isinstance(column, pl.Series):
            return ~column.is_in(values)
        elif isinstance(column, pl.Expr):
            return ~column.is_in(values)
        else:  # pandas Series
            return ~column.isin(values)

    @staticmethod
    def list_in(column1: Union[pl.Series, pl.Expr, pd.Series], 
                column2: Union[pl.Series, pl.Expr, pd.Series, str]) -> Union[pl.Series, pl.Expr, pd.Series]:
        """
        Checks if all elements in column2 are contained in column1.
        Both columns contain comma-separated strings like "A,B,C,D".
        
        If column2 is a single string, it will be compared against all rows in column1.
        Returns a boolean series where True indicates all items in column2 are present in column1.
        """
        ColumnComparator._validate_first_arg_is_valid(column1)

        # Helper function to convert comma-separated string to set
        def to_set(x):
            return set(item.strip() for item in x.split(','))

        # Handle scalar case (string comparison against whole series)
        if isinstance(column2, str):
            column2_set = to_set(column2)

            # If using Polars
            if isinstance(column1, pl.Series):
                return column1.apply(lambda x: column2_set.issubset(to_set(x)))
            elif isinstance(column1, pl.Expr):
                return column1.map_elements(lambda x: column2_set.issubset(to_set(x)))
            # If using Pandas
            elif isinstance(column1, pd.Series):
                return column1.apply(lambda x: column2_set.issubset(to_set(x)))

        # Handle series to series comparison (must be same length)
        else:
            ColumnComparator._validate_compatible_types(column1, column2)

            # If using Polars
            if isinstance(column1, pl.Series) and isinstance(column2, pl.Series):
                return pl.Series([
                    to_set(col2).issubset(to_set(col1))
                    for col1, col2 in zip(column1, column2)
                ])
            # For Expr, this would be more complex and would need to be handled differently
            # If using Pandas
            elif isinstance(column1, pd.Series) and isinstance(column2, pd.Series):
                return pd.Series([
                    to_set(col2).issubset(to_set(col1))
                    for col1, col2 in zip(column1, column2)
                ])

        raise TypeError("Unsupported types for list_in comparison")

    @staticmethod
    def _validate_first_arg_is_valid(column):
        """Validate that the first argument is a valid column representation (Series or Expression)."""
        if not isinstance(column, (pl.Series, pl.Expr, pd.Series)):
            raise ValueError("First argument must be a Polars Series, Polars Expression, or Pandas Series")

    @staticmethod
    def _validate_compatible_types(column1, column2):
        """Validate that the columns are compatible types and don't mix polars and pandas."""
        # Skip validation for scalar second arguments
        if not isinstance(column2, (pl.Series, pl.Expr, pd.Series)):
            return

        # Check for incompatible combinations
        if (isinstance(column1, pd.Series) and isinstance(column2, (pl.Series, pl.Expr))) or \
           (isinstance(column1, (pl.Series, pl.Expr)) and isinstance(column2, pd.Series)):
            raise TypeError("Cannot compare Polars and Pandas objects together")


# Helper function to get column from different dataframe types
def get_column(df, column_name: str) -> Union[pl.Series, pl.Expr, pd.Series]:
    """
    Helper function to get a column from different dataframe types.
    
    Args:
        df: DataFrame object (pl.DataFrame, pl.LazyFrame, or pd.DataFrame)
        column_name: Name of the column to access
        
    Returns:
        Column representation appropriate for the dataframe type
    """
    if isinstance(df, pl.LazyFrame):
        return pl.col(column_name)
    elif isinstance(df, pl.DataFrame):
        return df[column_name]
    elif isinstance(df, pd.DataFrame):
        return df[column_name]
    else:
        raise ValueError(f"Unsupported DataFrame type: {type(df)}")

def list_in(column1: Union[pl.Series, pl.Expr], column2: Union[pl.Series, pl.Expr, str]) -> pl.Expr:
        """
        Checks if all elements in column2 are contained in column1.
        Both columns contain comma-separated strings like "A,B,C,D".

        If column2 is a single string, it will be compared against all rows in column1.
        Returns a Boolean expression for use in a LazyFrame.
        """
        # Convert column1 and column2 to expressions if they are not already
        if isinstance(column1, pl.Series):
            column1 = pl.lit(column1)
        if isinstance(column2, pl.Series):
            column2 = pl.lit(column2)

        # Convert strings to list (split by ",")
        col1_list = column1.str.split(",")
        
        if isinstance(column2, str):  
            # If column2 is a single string, convert it to a list and check for subset
            col2_list = pl.lit(column2).str.split(",")
        else:
            # If column2 is an expression, split its values into lists
            col2_list = column2.str.split(",")

        # Check if all elements of col2_list are in col1_list
        return col2_list.list.eval(pl.element().is_in(col1_list)).list.all()


# Example usage
def example():
    # Create a Polars LazyFrame
    pldf = pl.scan_parquet("path/to/file.parquet")
    
    # For LazyFrame, use expressions
    amount_expr = pl.col("amount")
    truth_series = ColumnComparator.gt(amount_expr, 50)
    
    # Apply filter on the LazyFrame
    filtered_lf = pldf.filter(truth_series)
    
    # Now collect the result
    filtered_df = filtered_lf.collect()
    print(filtered_df)
    
    # Alternative: working with an already collected DataFrame
    df = pldf.collect()
    amount_series = df["amount"]
    truth_series = ColumnComparator.gt(amount_series, 50)
    filtered_pdf = df.filter(truth_series)
    print(filtered_pdf)


if __name__ == "__main__":
    example()