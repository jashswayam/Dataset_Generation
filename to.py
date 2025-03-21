from __future__ import annotations
from typing import Union, Any, Sequence
import polars as pl
import pandas as pd
import psutil


class ColumnComparator:
    """
    Extension methods for Polars and Pandas DataFrames to generate comparison expressions.
    These expressions can be used in `.filter()`, `.select()`, `.with_columns()`, etc.
    Works with both eager and lazy Polars DataFrames.
    """

    @staticmethod
    def gt(column1: Union[pl.Series, pd.Series, str], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 > column2."""
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a > b)

    @staticmethod
    def lt(column1: Union[pl.Series, pd.Series, str], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 < column2."""
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a < b)

    @staticmethod
    def gte(column1: Union[pl.Series, pd.Series, str], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 >= column2."""
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a >= b)

    @staticmethod
    def lte(column1: Union[pl.Series, pd.Series, str], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 <= column2."""
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a <= b)

    @staticmethod
    def eq(column1: Union[pl.Series, pd.Series, str], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 == column2."""
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a == b)

    @staticmethod
    def ne(column1: Union[pl.Series, pd.Series, str], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 != column2."""
        return ColumnComparator._compare_columns(column1, column2, lambda a, b: a != b)

    @staticmethod
    def contains(column: Union[pl.Series, pd.Series, str], pattern: str) -> pl.Expr:
        """Return an expression checking if a column contains a pattern."""
        col_expr = ColumnComparator._get_column_expr(column)
        return col_expr.str.contains(pattern)

    @staticmethod
    def not_contains(column: Union[pl.Series, pd.Series, str], pattern: str) -> pl.Expr:
        """Return an expression checking if a column does not contain a pattern."""
        col_expr = ColumnComparator._get_column_expr(column)
        return ~col_expr.str.contains(pattern)

    @staticmethod
    def is_in(column: Union[pl.Series, pd.Series, str], values: Sequence[Any]) -> pl.Expr:
        """Return an expression checking if column values are in a given sequence."""
        col_expr = ColumnComparator._get_column_expr(column)
        return col_expr.is_in(values)

    @staticmethod
    def not_in(column: Union[pl.Series, pd.Series, str], values: Sequence[Any]) -> pl.Expr:
        """Return an expression checking if column values are NOT in a given sequence."""
        col_expr = ColumnComparator._get_column_expr(column)
        return ~col_expr.is_in(values)

    @staticmethod
    def _compare_columns(column1: Union[pl.Series, pd.Series, str], 
                         column2: Union[str, pl.Expr, Any], 
                         comparison_func) -> pl.Expr:
        """Helper method to handle different column comparison scenarios."""
        col1_expr = ColumnComparator._get_column_expr(column1)
        col2_expr = ColumnComparator._get_column_expr(column2)
        return comparison_func(col1_expr, col2_expr)

    @staticmethod
    def _get_column_expr(column: Union[pl.Series, pd.Series, str, Any]) -> pl.Expr:
        """Convert various input types to Polars expressions."""
        if isinstance(column, str):  # Column name
            return pl.col(column)
        elif isinstance(column, pl.Series):  # Polars Series
            return pl.col(column.name)
        elif isinstance(column, pd.Series):  # Pandas Series
            return pl.col(column.name)
        elif isinstance(column, (int, float, bool)):  # Handle scalar values
            return pl.lit(column)
        else:
            # Try to handle other types as literals
            try:
                return pl.lit(column)
            except:
                raise TypeError("Column should be a column name, Polars Series, Pandas Series, or scalar value")


# Utility functions for parquet handling with memory tracking
def load_parquet(file_path: str, lazy: bool = False) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Load a parquet file into either an eager DataFrame or a LazyFrame with memory tracking.
    
    Args:
        file_path: Path to the parquet file
        lazy: If True, return a LazyFrame; otherwise, return an eager DataFrame
        
    Returns:
        Polars DataFrame or LazyFrame
    """
    # Get memory usage before loading
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Load the dataframe
    if lazy:
        df = pl.scan_parquet(file_path)
    else:
        df = pl.read_parquet(file_path)
    
    # Get memory usage after loading
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Print memory usage information
    print(f"Memory before loading: {mem_before:.2f} MB")
    print(f"Memory after loading: {mem_after:.2f} MB")
    print(f"Memory used for loading: {mem_after - mem_before:.2f} MB")
    
    return df


# Example usage
def example():
    """Example showing basic usage with memory tracking"""
    import tempfile
    
    # Create a sample dataframe and save as parquet
    pldf = pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": ["apple", "banana", "cherry", "date", "elderberry"]
    })
    
    # Create a temporary file for the parquet
    with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
        pldf.write_parquet(tmp.name)
        print(f"Saved sample data to temporary parquet file: {tmp.name}")
        
        # Memory tracking during eager loading
        print("\n=== EAGER LOADING ===")
        eager_df = load_parquet(tmp.name, lazy=False)
        print(f"Loaded eager DataFrame with shape: {eager_df.shape}")
        
        # Apply filter using ColumnComparator
        print("\n=== EAGER FILTERING ===")
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        filtered_df = eager_df.filter(ColumnComparator.gt("A", 2))
        
        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"Memory used for filtering: {mem_after - mem_before:.2f} MB")
        print(f"Filtered DataFrame with {len(filtered_df)} rows:")
        print(filtered_df)
        
        # Memory tracking during lazy loading
        print("\n=== LAZY LOADING ===")
        lazy_df = load_parquet(tmp.name, lazy=True)
        print(f"Loaded lazy DataFrame with schema: {lazy_df.schema}")
        
        # Apply filter using ColumnComparator with lazy evaluation
        print("\n=== LAZY FILTERING ===")
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        filtered_lazy = lazy_df.filter(ColumnComparator.gt("A", 2))
        result = filtered_lazy.collect()
        
        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"Memory used for lazy filtering and collection: {mem_after - mem_before:.2f} MB")
        print(f"Filtered DataFrame with {len(result)} rows:")
        print(result)
        
        # Chains of operations with lazy evaluation
        print("\n=== LAZY CHAIN OPERATIONS ===")
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        result2 = (
            pl.scan_parquet(tmp.name)
            .filter(ColumnComparator.gt("A", 2))
            .filter(ColumnComparator.contains("C", "e"))
            .select(["A", "C"])
            .collect()
        )
        
        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"Memory used for chain operations: {mem_after - mem_before:.2f} MB")
        print("Chained lazy operations result:")
        print(result2)


if __name__ == "__main__":
    # Run the example
    example()