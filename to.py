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

    
        def list_in(column1: Union[pl.Series, pl.Expr, pd.Series], 
           column2: Union[pl.Series, pl.Expr, pd.Series, str]) -> Union[pl.Series, pl.Expr, pd.Series]:
    """
    Checks if all elements in column2 are contained in column1.
    Both columns contain comma-separated strings like "A,B,C,D".
    
    If column2 is a single string, it will be compared against all rows in column1.
    Returns a boolean series where True indicates column2 is contained in column1.
    """
    ColumnComparator._validate_first_arg_is_valid(column1)
    
    # Handle scalar case (string comparison against whole series)
    if isinstance(column2, str):
        # If using Polars
        if isinstance(column1, (pl.Series, pl.Expr)):
            return column1.apply(lambda x: all(item.strip() in [i.strip() for i in x.split(',')] 
                                              for item in column2.split(',')))
        # If using Pandas
        elif isinstance(column1, pd.Series):
            return column1.apply(lambda x: all(item.strip() in [i.strip() for i in x.split(',')]
                                              for item in column2.split(',')))
    
    # Handle series to series comparison (must be same length)
    else:
        ColumnComparator._validate_compatible_types(column1, column2)
        
        # If using Polars
        if isinstance(column1, (pl.Series, pl.Expr)):
            return pl.Series([
                all(item.strip() in [i.strip() for i in col1.split(',')]
                    for item in col2.split(','))
                for col1, col2 in zip(column1, column2)
            ])
        # If using Pandas
        elif isinstance(column1, pd.Series):
            return pd.Series([
                all(item.strip() in [i.strip() for i in col1.split(',')]
                    for item in col2.split(','))
                for col1, col2 in zip(column1, column2)
            ])
    
    raise TypeError("Unsupported types for list_in comparison")
        if isinstance(column, pl.Series):
            return ~column.is_in(values)
        elif isinstance(column, pl.Expr):
            return ~column.is_in(values)
        else:  # pandas Series
            return ~column.isin(values)

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
    
    # Example with polars LazyFrame
    lazy_df = pldf.lazy()

    # Using with pandas - direct filter
    print("\nPandas: Rows where A > 3:")
    truth_series = ColumnComparator.gt(get_column(pdf, "A"), 3)
    filtered_pdf = pdf[truth_series]
    print(filtered_pdf)

    # Using with polars DataFrame - direct filter
    print("\nPolars DataFrame: Rows where A > B/10:")
    truth_series = ColumnComparator.gt(get_column(pldf, "A"), get_column(pldf, "B")/10)
    filtered_pldf = pldf.filter(truth_series)
    print(filtered_pldf)

    # Using with polars LazyFrame - expression-based filter
    print("\nPolars LazyFrame: Rows where A > 3:")
    expr = ColumnComparator.gt(get_column(lazy_df, "A"), 3)
    filtered_lazy = lazy_df.filter(expr)
    filtered_result = filtered_lazy.collect()  # Executes the lazy query
    print(filtered_result)

    # String operation example with polars LazyFrame
    print("\nPolars LazyFrame: Rows where C contains 'a':")
    expr = ColumnComparator.contains(get_column(lazy_df, "C"), "a")
    filtered_lazy = lazy_df.filter(expr)
    filtered_result = filtered_lazy.collect()
    print(filtered_result)

    # Multiple conditions example with polars LazyFrame
    print("\nPolars LazyFrame: Rows where A > 2 AND C contains 'e':")
    condition1 = ColumnComparator.gt(get_column(lazy_df, "A"), 2)
    condition2 = ColumnComparator.contains(get_column(lazy_df, "C"), "e")
    filtered_lazy = lazy_df.filter(condition1 & condition2)
    filtered_result = filtered_lazy.collect()
    print(filtered_result)

    # This should raise an error (scalar as first argument)
    try:
        print("\nThis should fail - scalar as first argument:")
        ColumnComparator.gte(40, get_column(pdf, "B"))
    except ValueError as e:
        print(f"Correctly caught error: {e}")


if __name__ == "__main__":
    example()