from __future__ import annotations
from typing import Union, Any, Sequence
import polars as pl
import pandas as pd


class ColumnComparator:
    """
    Extension methods for Polars and Pandas DataFrames to generate comparison expressions.
    These expressions can be used in `.filter()`, `.select()`, `.with_columns()`, etc.
    """

    @staticmethod
    def gt(column1: Union[str, pl.Series, pd.Series], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 > column2."""
        ColumnComparator._validate_first_arg_is_column(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a > b)

    @staticmethod
    def lt(column1: Union[str, pl.Series, pd.Series], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 < column2."""
        ColumnComparator._validate_first_arg_is_column(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a < b)

    @staticmethod
    def gte(column1: Union[str, pl.Series, pd.Series], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 >= column2."""
        ColumnComparator._validate_first_arg_is_column(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a >= b)

    @staticmethod
    def lte(column1: Union[str, pl.Series, pd.Series], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 <= column2."""
        ColumnComparator._validate_first_arg_is_column(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a <= b)

    @staticmethod
    def eq(column1: Union[str, pl.Series, pd.Series], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 == column2."""
        ColumnComparator._validate_first_arg_is_column(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a == b)

    @staticmethod
    def ne(column1: Union[str, pl.Series, pd.Series], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 != column2."""
        ColumnComparator._validate_first_arg_is_column(column1)
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a != b)

    @staticmethod
    def contains(column: Union[str, pl.Series, pd.Series], pattern: str) -> pl.Expr:
        """Return an expression checking if a column contains a pattern."""
        ColumnComparator._validate_first_arg_is_column(column)
        col_expr = ColumnComparator._get_column_expr(column)
        return col_expr.str.contains(pattern)

    @staticmethod
    def not_contains(column: Union[str, pl.Series, pd.Series], pattern: str) -> pl.Expr:
        """Return an expression checking if a column does not contain a pattern."""
        ColumnComparator._validate_first_arg_is_column(column)
        col_expr = ColumnComparator._get_column_expr(column)
        return ~col_expr.str.contains(pattern)

    @staticmethod
    def is_in(column: Union[str, pl.Series, pd.Series], values: Sequence[Any]) -> pl.Expr:
        """Return an expression checking if column values are in a given sequence."""
        ColumnComparator._validate_first_arg_is_column(column)
        col_expr = ColumnComparator._get_column_expr(column)
        return col_expr.is_in(values)

    @staticmethod
    def not_in(column: Union[str, pl.Series, pd.Series], values: Sequence[Any]) -> pl.Expr:
        """Return an expression checking if column values are NOT in a given sequence."""
        ColumnComparator._validate_first_arg_is_column(column)
        col_expr = ColumnComparator._get_column_expr(column)
        return ~col_expr.is_in(values)

    @staticmethod
    def _validate_first_arg_is_column(column):
        """Validate that the first argument is a column reference (not a scalar)."""
        if ColumnComparator._is_scalar(column):
            raise ValueError("First argument must be a column Series, not a scalar value")

    @staticmethod
    def _compare_columns(column1: Union[str, pl.Series, pd.Series], 
                         column2: Union[str, pl.Expr, Any], 
                         comparison_func) -> pl.Expr:
        """Helper method to handle different column comparison scenarios."""
        col1_expr = ColumnComparator._get_column_expr(column1)
        col2_expr = ColumnComparator._get_column_expr(column2)
        return comparison_func(col1_expr, col2_expr)

    @staticmethod
    def _is_scalar(value):
        """Check if a value is a scalar (not a column reference)."""
        return (isinstance(value, (int, float, bool)) or 
                (isinstance(value, str) and value.startswith("'") and value.endswith("'")))

    @staticmethod
    def _get_column_expr(column: Union[str, pl.Series, pd.Series, Any]) -> pl.Expr:
        """Convert various input types to Polars expressions."""
        if isinstance(column, str):  # Column name
            return pl.col(column)
        elif isinstance(column, pl.Series):  # Polars Series
            return pl.col(column.name)
        elif isinstance(column, pd.Series):  # Pandas Series
            return pl.col(column.name)
        elif isinstance(column, dict) and len(column) == 1:  # Dict with DataFrame and column name
            # For cases like {"df": "column_name"}
            df_key = list(column.keys())[0]
            col_name = column[df_key]
            return pl.col(col_name)
        elif isinstance(column, (int, float, bool)):  # Handle scalar values
            return pl.lit(column)
        elif isinstance(column, pl.DataFrame) and len(column.columns) == 1:
            # Single column DataFrame
            return pl.col(column.columns[0])
        elif isinstance(column, pd.DataFrame) and len(column.columns) == 1:
            # Single column DataFrame
            return pl.col(column.columns[0])
        else:
            raise TypeError("Column should be a column name, Polars Series, Pandas Series, or scalar value")


# Extend DataFrame classes with filter_by method
def extend_dataframes():
    """Extend Polars and Pandas DataFrames with filter_by method."""
    
    # For Polars DataFrame
    def polars_filter_by(self, expr):
        return self.filter(expr)
    
    # For Pandas DataFrame
    def pandas_filter_by(self, expr):
        # Convert pandas to polars, apply filter, convert back
        pl_df = pl.from_pandas(self)
        filtered_pl_df = pl_df.filter(expr)
        return filtered_pl_df.to_pandas()
    
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

    # Using column name directly with pandas
    print("\nPandas: Rows where A > 3:")
    filtered_pdf = pdf.filter_by(ColumnComparator.gt("A", 3))
    print(filtered_pdf)
    
    # Using column name directly with polars
    print("\nPolars: Rows where A > 3:")
    filtered_pldf = pldf.filter_by(ColumnComparator.gt("A", "B"))
    print(filtered_pldf)
    
   
    
    # These should work fine (column as first argument)
    print("\nRows where B >= 30:")
    print(pdf.filter_by(ColumnComparator.gte(40, 30)))


if __name__ == "__main__":
    example()
