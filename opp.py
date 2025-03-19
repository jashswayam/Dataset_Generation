from __future__ import annotations
from typing import Sequence, Any, List, Union, Optional
import polars as pl


class ColumnComparator:
    """
    A class that implements various comparison operations for Polars DataFrame columns
    using static methods only, no object instantiation required.
    """
    
    @staticmethod
    def _get_expr(column: Union[pl.Series, str]) -> pl.Expr:
        """
        Convert column to a Polars expression.
        
        Args:
            column: Column as string name or Polars Series
            
        Returns:
            Polars expression
        """
        if isinstance(column, str):
            return pl.col(column)
        return column
    
    @staticmethod
    def _process_expressions(expressions: List[pl.Expr], mode: str = "any") -> pl.Expr:
        """
        Process multiple expressions with the given mode.
        
        Args:
            expressions: List of expressions to combine
            mode: 'any' or 'all' (default: 'any')
            
        Returns:
            Combined Polars expression
        """
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    @staticmethod
    def greater_than_equal_to(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """
        Check if column(s) is/are greater than or equal to another value.
        
        Args:
            columns: Column name(s) or Polars Series
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        if not isinstance(columns, list):
            columns = [columns]
            
        expressions = [ColumnComparator._get_expr(col) >= other for col in columns]
        return ColumnComparator._process_expressions(expressions, mode)
    
    @staticmethod
    def greater_than(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """
        Check if column(s) is/are greater than another value.
        
        Args:
            columns: Column name(s) or Polars Series
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        if not isinstance(columns, list):
            columns = [columns]
            
        expressions = [ColumnComparator._get_expr(col) > other for col in columns]
        return ColumnComparator._process_expressions(expressions, mode)
    
    @staticmethod
    def less_than_equal_to(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """
        Check if column(s) is/are less than or equal to another value.
        
        Args:
            columns: Column name(s) or Polars Series
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        if not isinstance(columns, list):
            columns = [columns]
            
        expressions = [ColumnComparator._get_expr(col) <= other for col in columns]
        return ColumnComparator._process_expressions(expressions, mode)
    
    @staticmethod
    def less_than(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """
        Check if column(s) is/are less than another value.
        
        Args:
            columns: Column name(s) or Polars Series
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        if not isinstance(columns, list):
            columns = [columns]
            
        expressions = [ColumnComparator._get_expr(col) < other for col in columns]
        return ColumnComparator._process_expressions(expressions, mode)
    
    @staticmethod
    def equals(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """
        Check if column(s) is/are equal to another value.
        
        Args:
            columns: Column name(s) or Polars Series
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        if not isinstance(columns, list):
            columns = [columns]
            
        expressions = [ColumnComparator._get_expr(col) == other for col in columns]
        return ColumnComparator._process_expressions(expressions, mode)
    
    @staticmethod
    def not_equals(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """
        Check if column(s) is/are not equal to another value.
        
        Args:
            columns: Column name(s) or Polars Series
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        if not isinstance(columns, list):
            columns = [columns]
            
        expressions = [ColumnComparator._get_expr(col) != other for col in columns]
        return ColumnComparator._process_expressions(expressions, mode)
    
    @staticmethod
    def contains(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        pattern: str,
        mode: str = "any"
    ) -> pl.Expr:
        """
        Check if string column(s) contain(s) a specific pattern.
        
        Args:
            columns: Column name(s) or Polars Series (must be string columns)
            pattern: Pattern to search for
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        if not isinstance(columns, list):
            columns = [columns]
            
        expressions = [ColumnComparator._get_expr(col).str.contains(pattern) for col in columns]
        return ColumnComparator._process_expressions(expressions, mode)
    
    @staticmethod
    def not_contains(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        pattern: str,
        mode: str = "any"
    ) -> pl.Expr:
        """
        Check if string column(s) do(es) not contain a specific pattern.
        
        Args:
            columns: Column name(s) or Polars Series (must be string columns)
            pattern: Pattern to search for
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        if not isinstance(columns, list):
            columns = [columns]
            
        expressions = [~ColumnComparator._get_expr(col).str.contains(pattern) for col in columns]
        return ColumnComparator._process_expressions(expressions, mode)
    
    @staticmethod
    def is_in(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        values: Sequence[Any],
        mode: str = "any"
    ) -> pl.Expr:
        """
        Check if column(s) value(s) are in a sequence of values.
        
        Args:
            columns: Column name(s) or Polars Series
            values: Sequence of values to check against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        if not isinstance(columns, list):
            columns = [columns]
            
        expressions = [ColumnComparator._get_expr(col).is_in(values) for col in columns]
        return ColumnComparator._process_expressions(expressions, mode)
    
    @staticmethod
    def not_in(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        values: Sequence[Any],
        mode: str = "any"
    ) -> pl.Expr:
        """
        Check if column(s) value(s) are not in a sequence of values.
        
        Args:
            columns: Column name(s) or Polars Series
            values: Sequence of values to check against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        if not isinstance(columns, list):
            columns = [columns]
            
        expressions = [~ColumnComparator._get_expr(col).is_in(values) for col in columns]
        return ColumnComparator._process_expressions(expressions, mode)
    
    # Aliases for better readability
    @staticmethod
    def gte(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """Alias for greater_than_equal_to"""
        return ColumnComparator.greater_than_equal_to(columns, other, mode)
    
    @staticmethod
    def gt(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """Alias for greater_than"""
        return ColumnComparator.greater_than(columns, other, mode)
    
    @staticmethod
    def lte(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """Alias for less_than_equal_to"""
        return ColumnComparator.less_than_equal_to(columns, other, mode)
    
    @staticmethod
    def lt(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """Alias for less_than"""
        return ColumnComparator.less_than(columns, other, mode)
    
    @staticmethod
    def eq(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """Alias for equals"""
        return ColumnComparator.equals(columns, other, mode)
    
    @staticmethod
    def ne(
        columns: Union[str, pl.Series, List[Union[str, pl.Series]]],
        other: Any,
        mode: str = "any"
    ) -> pl.Expr:
        """Alias for not_equals"""
        return ColumnComparator.not_equals(columns, other, mode)


# Example usage with the static methods
def example():
    # Create a sample DataFrame
    df = pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": ["apple", "banana", "cherry", "date", "elderberry"]
    })
    
    # Example 1: Filter rows where column A > 3
    filtered_df = df.filter(ColumnComparator.greater_than("A", 3))
    print("Rows where A > 3:")
    print(filtered_df)
    
    # Example 2: Filter rows where column B is in [20, 40]
    filtered_df = df.filter(ColumnComparator.is_in("B", [20, 40]))
    print("\nRows where B is in [20, 40]:")
    print(filtered_df)
    
    # Example 3: Filter rows where column C contains "a"
    filtered_df = df.filter(ColumnComparator.contains("C", "a"))
    print("\nRows where C contains 'a':")
    print(filtered_df)
    
    # Example 4: Filter rows where either A > 3 or B > 30
    filtered_df = df.filter(ColumnComparator.greater_than(["A", "B"], 3))  # default mode is "any"
    print("\nRows where either A > 3 or B > 3:")
    print(filtered_df)
    
    # Example 5: Filter rows where all numeric columns are less than 50
    filtered_df = df.filter(ColumnComparator.less_than(["A", "B"], 50, mode="all"))
    print("\nRows where all numeric columns are < 50:")
    print(filtered_df)
    
    # Example 6: Using shorter aliases and combining expressions
    filtered_df = df.filter(
        ColumnComparator.gt(["A", "B"], 3) & ColumnComparator.lt(["A", "B"], 40, mode="all")
    )
    print("\nRows where (A > 3 OR B > 3) AND (A < 40 AND B < 40):")
    print(filtered_df)


def example_with_parquet():
    # Read the parquet file
    df = pl.read_parquet("grouped_data.parquet")
    
    # Display basic information about the DataFrame
    print("DataFrame schema:")
    print(df.schema)
    print("\nFirst 5 rows:")
    print(df.head(5))
    
    # Example 1: Basic filtering on a single column
    # Let's assume the parquet file has columns like 'group', 'value', 'category'
    filtered_df = df.filter(ColumnComparator.gt("value", 100))
    print("\nRows where value > 100:")
    print(filtered_df.head(5))
    
    # Example 2: Filtering with multiple conditions
    filtered_df = df.filter(
        ColumnComparator.is_in("group", ["A", "B"]) & ColumnComparator.gt("value", 50)
    )
    print("\nRows where group is either 'A' or 'B' AND value > 50:")
    print(filtered_df.head(5))
    
    # Example 3: Using the multi-column approach
    filtered_df = df.filter(ColumnComparator.gt(["value1", "value2"], 75, mode="any"))
    print("\nRows where either value1 > 75 OR value2 > 75:")
    print(filtered_df.head(5))
    
    # Example 4: Combining different comparators
    filtered_df = df.filter(
        ColumnComparator.gt("value", 50) & ColumnComparator.contains("category", "important")
    )
    print("\nRows where value > 50 AND category contains 'important':")
    print(filtered_df.head(5))
    
    # Example 5: Advanced filtering with aggregation
    # Get groups where the average value exceeds 100
    high_avg_groups = df.group_by("group").agg(pl.col("value").mean().alias("avg_value")).filter(
        ColumnComparator.gt("avg_value", 100)
    ).select("group")
    
    # Then filter the original DataFrame to only include those groups
    filtered_df = df.filter(
        ColumnComparator.is_in("group", high_avg_groups["group"])
    )
    print("\nRows belonging to groups with average value > 100:")
    print(filtered_df.head(5))


if __name__ == "__main__":
    example()
    # Uncomment to run the parquet example
    # example_with_parquet()