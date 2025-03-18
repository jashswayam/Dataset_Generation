from __future__ import annotations
from typing import Sequence, Any, TypeVar, Dict, Tuple, List, Optional, Union
import polars as pl

T = TypeVar('T')  # Generic type variable

class ColumnComparator:
    """
    A class that implements various comparison operations for Polars DataFrame columns.
    Works with both single and multiple columns.
    """
    def __init__(self, columns: Union[pl.Series, str, List[Union[pl.Series, str]]]):
        """
        Initialize a comparator for Polars column(s).
        
        Args:
            columns: Either a single column (Polars Series or string) or multiple columns (list)
        """
        if isinstance(columns, list):
            self.columns = columns
            self.is_multi = True
        else:
            self.columns = [columns]
            self.is_multi = False
    
    def _get_expr(self, column: Union[pl.Series, str]) -> pl.Expr:
        """
        Convert column to a Polars expression.
        """
        if isinstance(column, str):
            return pl.col(column)
        return column
    
    def greater_than_equal_to(self, other: Any, mode: str = "any") -> pl.Expr:
        """
        Check if column(s) is/are greater than or equal to another value.
        
        Args:
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
        
        Returns:
            Polars expression
        """
        expressions = [self._get_expr(col) >= other for col in self.columns]
        
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    def greater_than(self, other: Any, mode: str = "any") -> pl.Expr:
        """
        Check if column(s) is/are greater than another value.
        
        Args:
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
        
        Returns:
            Polars expression
        """
        expressions = [self._get_expr(col) > other for col in self.columns]
        
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    def less_than_equal_to(self, other: Any, mode: str = "any") -> pl.Expr:
        """
        Check if column(s) is/are less than or equal to another value.
        
        Args:
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
        
        Returns:
            Polars expression
        """
        expressions = [self._get_expr(col) <= other for col in self.columns]
        
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    def less_than(self, other: Any, mode: str = "any") -> pl.Expr:
        """
        Check if column(s) is/are less than another value.
        
        Args:
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
        
        Returns:
            Polars expression
        """
        expressions = [self._get_expr(col) < other for col in self.columns]
        
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    def equals(self, other: Any, mode: str = "any") -> pl.Expr:
        """
        Check if column(s) is/are equal to another value.
        
        Args:
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
        
        Returns:
            Polars expression
        """
        expressions = [self._get_expr(col) == other for col in self.columns]
        
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    def not_equals(self, other: Any, mode: str = "any") -> pl.Expr:
        """
        Check if column(s) is/are not equal to another value.
        
        Args:
            other: Value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
        
        Returns:
            Polars expression
        """
        expressions = [self._get_expr(col) != other for col in self.columns]
        
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    def contains(self, pattern: str, mode: str = "any") -> pl.Expr:
        """
        Check if string column(s) contain(s) a specific pattern.
        
        Args:
            pattern: Pattern to search for
            mode: For multiple columns, 'any' or 'all' (default: 'any')
        
        Returns:
            Polars expression
        """
        expressions = [self._get_expr(col).str.contains(pattern) for col in self.columns]
        
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    def not_contains(self, pattern: str, mode: str = "any") -> pl.Expr:
        """
        Check if string column(s) do(es) not contain a specific pattern.
        
        Args:
            pattern: Pattern to search for
            mode: For multiple columns, 'any' or 'all' (default: 'any')
        
        Returns:
            Polars expression
        """
        expressions = [~self._get_expr(col).str.contains(pattern) for col in self.columns]
        
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    def is_in(self, values: Sequence[Any], mode: str = "any") -> pl.Expr:
        """
        Check if column(s) value(s) are in a sequence of values.
        
        Args:
            values: Sequence of values to check against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
        
        Returns:
            Polars expression
        """
        expressions = [self._get_expr(col).is_in(values) for col in self.columns]
        
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    def not_in(self, values: Sequence[Any], mode: str = "any") -> pl.Expr:
        """
        Check if column(s) value(s) are not in a sequence of values.
        
        Args:
            values: Sequence of values to check against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
        
        Returns:
            Polars expression
        """
        expressions = [~self._get_expr(col).is_in(values) for col in self.columns]
        
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)
    
    # Aliases for better readability
    def gte(self, other: Any, mode: str = "any") -> pl.Expr:
        """Alias for greater_than_equal_to"""
        return self.greater_than_equal_to(other, mode)
    
    def gt(self, other: Any, mode: str = "any") -> pl.Expr:
        """Alias for greater_than"""
        return self.greater_than(other, mode)
    
    def lte(self, other: Any, mode: str = "any") -> pl.Expr:
        """Alias for less_than_equal_to"""
        return self.less_than_equal_to(other, mode)
    
    def lt(self, other: Any, mode: str = "any") -> pl.Expr:
        """Alias for less_than"""
        return self.less_than(other, mode)
    
    def eq(self, other: Any, mode: str = "any") -> pl.Expr:
        """Alias for equals"""
        return self.equals(other, mode)
    
    def ne(self, other: Any, mode: str = "any") -> pl.Expr:
        """Alias for not_equals"""
        return self.not_equals(other, mode)


# Example usage
def example():
    # Create a sample DataFrame
    df = pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": ["apple", "banana", "cherry", "date", "elderberry"]
    })
    
    # Example 1: Filter rows where column A > 3
    comp = ColumnComparator("A")
    filtered_df = df.filter(comp.greater_than(3))
    print("Rows where A > 3:")
    print(filtered_df)
    
    # Example 2: Filter rows where column B is in [20, 40]
    comp = ColumnComparator("B")
    filtered_df = df.filter(comp.is_in([20, 40]))
    print("\nRows where B is in [20, 40]:")
    print(filtered_df)
    
    # Example 3: Filter rows where column C contains "a"
    comp = ColumnComparator("C")
    filtered_df = df.filter(comp.contains("a"))
    print("\nRows where C contains 'a':")
    print(filtered_df)
    
    # Example 4: Filter rows where either A > 3 or B > 30
    comp = ColumnComparator(["A", "B"])
    filtered_df = df.filter(comp.greater_than(3))  # default mode is "any"
    print("\nRows where either A > 3 or B > 3:")
    print(filtered_df)
    
    # Example 5: Filter rows where all numeric columns are less than 50
    comp = ColumnComparator(["A", "B"])
    filtered_df = df.filter(comp.less_than(50, mode="all"))
    print("\nRows where all numeric columns are < 50:")
    print(filtered_df)
    
    # Example 6: Using shorter aliases
    comp = ColumnComparator(["A", "B"])
    filtered_df = df.filter(comp.gt(3) & comp.lt(40, mode="all"))
    print("\nRows where (A > 3 OR B > 3) AND (A < 40 AND B < 40):")
    print(filtered_df)


if __name__ == "__main__":
    example()