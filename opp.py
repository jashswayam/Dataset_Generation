from __future__ import annotations
from typing import Sequence, Any, List, Union, Optional
import polars as pl


class ColumnComparator:
    """
    Extension methods for Polars DataFrame to simplify column comparisons.
    Each method is independent and statically defined.
    """

    @staticmethod
    def gt(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """
        Filter rows where column1 > column2.
        
        Args:
            df: Polars DataFrame
            column1: First column to compare (string name or expression)
            column2: Second column or value to compare against
            
        Returns:
            Filtered DataFrame
        """
        # Convert string column names to expressions
        if isinstance(column1, str):
            column1 = pl.col(column1)
        elif not isinstance(column1, pl.Expr):
            column1 = pl.lit(column1)
            
        if isinstance(column2, str):
            column2 = pl.col(column2)
        elif not isinstance(column2, pl.Expr):
            column2 = pl.lit(column2)
            
        return df.filter(column1 > column2)
    
    @staticmethod
    def lt(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """
        Filter rows where column1 < column2.
        
        Args:
            df: Polars DataFrame
            column1: First column to compare (string name or expression)
            column2: Second column or value to compare against
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(column1, str):
            column1 = pl.col(column1)
        elif not isinstance(column1, pl.Expr):
            column1 = pl.lit(column1)
            
        if isinstance(column2, str):
            column2 = pl.col(column2)
        elif not isinstance(column2, pl.Expr):
            column2 = pl.lit(column2)
            
        return df.filter(column1 < column2)
    
    @staticmethod
    def gte(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """
        Filter rows where column1 >= column2.
        
        Args:
            df: Polars DataFrame
            column1: First column to compare (string name or expression)
            column2: Second column or value to compare against
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(column1, str):
            column1 = pl.col(column1)
        elif not isinstance(column1, pl.Expr):
            column1 = pl.lit(column1)
            
        if isinstance(column2, str):
            column2 = pl.col(column2)
        elif not isinstance(column2, pl.Expr):
            column2 = pl.lit(column2)
            
        return df.filter(column1 >= column2)
    
    @staticmethod
    def lte(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """
        Filter rows where column1 <= column2.
        
        Args:
            df: Polars DataFrame
            column1: First column to compare (string name or expression)
            column2: Second column or value to compare against
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(column1, str):
            column1 = pl.col(column1)
        elif not isinstance(column1, pl.Expr):
            column1 = pl.lit(column1)
            
        if isinstance(column2, str):
            column2 = pl.col(column2)
        elif not isinstance(column2, pl.Expr):
            column2 = pl.lit(column2)
            
        return df.filter(column1 <= column2)
    
    @staticmethod
    def eq(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """
        Filter rows where column1 == column2.
        
        Args:
            df: Polars DataFrame
            column1: First column to compare (string name or expression)
            column2: Second column or value to compare against
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(column1, str):
            column1 = pl.col(column1)
        elif not isinstance(column1, pl.Expr):
            column1 = pl.lit(column1)
            
        if isinstance(column2, str):
            column2 = pl.col(column2)
        elif not isinstance(column2, pl.Expr):
            column2 = pl.lit(column2)
            
        return df.filter(column1 == column2)
    
    @staticmethod
    def ne(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """
        Filter rows where column1 != column2.
        
        Args:
            df: Polars DataFrame
            column1: First column to compare (string name or expression)
            column2: Second column or value to compare against
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(column1, str):
            column1 = pl.col(column1)
        elif not isinstance(column1, pl.Expr):
            column1 = pl.lit(column1)
            
        if isinstance(column2, str):
            column2 = pl.col(column2)
        elif not isinstance(column2, pl.Expr):
            column2 = pl.lit(column2)
            
        return df.filter(column1 != column2)
    
    @staticmethod
    def contains(df: pl.DataFrame, column: Union[str, pl.Expr], pattern: str) -> pl.DataFrame:
        """
        Filter rows where column contains pattern.
        
        Args:
            df: Polars DataFrame
            column: Column to check (string name or expression)
            pattern: Pattern to search for
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(column, str):
            column = pl.col(column)
        
        return df.filter(column.str.contains(pattern))
    
    @staticmethod
    def not_contains(df: pl.DataFrame, column: Union[str, pl.Expr], pattern: str) -> pl.DataFrame:
        """
        Filter rows where column does not contain pattern.
        
        Args:
            df: Polars DataFrame
            column: Column to check (string name or expression)
            pattern: Pattern to search for
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(column, str):
            column = pl.col(column)
        
        return df.filter(~column.str.contains(pattern))
    
    @staticmethod
    def is_in(df: pl.DataFrame, column: Union[str, pl.Expr], values: Sequence[Any]) -> pl.DataFrame:
        """
        Filter rows where column value is in a sequence of values.
        
        Args:
            df: Polars DataFrame
            column: Column to check (string name or expression)
            values: Sequence of values to check against
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(column, str):
            column = pl.col(column)
        
        return df.filter(column.is_in(values))
    
    @staticmethod
    def not_in(df: pl.DataFrame, column: Union[str, pl.Expr], values: Sequence[Any]) -> pl.DataFrame:
        """
        Filter rows where column value is not in a sequence of values.
        
        Args:
            df: Polars DataFrame
            column: Column to check (string name or expression)
            values: Sequence of values to check against
            
        Returns:
            Filtered DataFrame
        """
        if isinstance(column, str):
            column = pl.col(column)
        
        return df.filter(~column.is_in(values))
    
    # Longer aliases for better readability
    @staticmethod
    def greater_than(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """Alias for gt"""
        return ColumnComparator.gt(df, column1, column2)
    
    @staticmethod
    def less_than(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """Alias for lt"""
        return ColumnComparator.lt(df, column1, column2)
    
    @staticmethod
    def greater_than_equal_to(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """Alias for gte"""
        return ColumnComparator.gte(df, column1, column2)
    
    @staticmethod
    def less_than_equal_to(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """Alias for lte"""
        return ColumnComparator.lte(df, column1, column2)
    
    @staticmethod
    def equals(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """Alias for eq"""
        return ColumnComparator.eq(df, column1, column2)
    
    @staticmethod
    def not_equals(df: pl.DataFrame, column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.DataFrame:
        """Alias for ne"""
        return ColumnComparator.ne(df, column1, column2)


# Example usage
def example():
    # Create a sample DataFrame
    df = pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": ["apple", "banana", "cherry", "date", "elderberry"]
    })

    # Example 1: Using the new syntax to filter rows where A > 3
    filtered_df = ColumnComparator.gt(df, "A", 3)
    print("Rows where A > 3:")
    print(filtered_df)

    # Example 2: Filter rows where A > B (column to column comparison)
    filtered_df = ColumnComparator.gt(df, "A", "B")
    print("\nRows where A > B:")
    print(filtered_df)

    # Example 3: Filter rows where C contains 'a'
    filtered_df = ColumnComparator.contains(df, "C", "a")
    print("\nRows where C contains 'a':")
    print(filtered_df)

    # Example 4: Filter rows where B is in [20, 40]
    filtered_df = ColumnComparator.is_in(df, "B", [20, 40])
    print("\nRows where B is in [20, 40]:")
    print(filtered_df)

    # Example 5: Complex filtering with chaining
    filtered_df = ColumnComparator.lt(ColumnComparator.gt(df, "A", 2), "B", 40)
    print("\nRows where A > 2 AND B < 40:")
    print(filtered_df)

    # Example 6: Using expressions
    filtered_df = ColumnComparator.gt(df, pl.col("A") + 1, pl.col("B") / 10)
    print("\nRows where (A + 1) > (B/10):")
    print(filtered_df)


if __name__ == "__main__":
    example()