from __future__ import annotations
from typing import Sequence, Any, List, Union, Optional
import polars as pl


class ColumnComparator:
    """
    Extension methods for Polars DataFrame to simplify column comparisons.
    This class adds methods to the DataFrame class to compare columns with values or other columns.
    """
    
    @staticmethod
    def _install_methods():
        """
        Install the comparison methods on the pl.DataFrame class.
        """
        def _make_method(method_name, doc_string):
            def method(df, column1, column2, mode="any"):
                return df.filter(getattr(ColumnComparator, method_name)(column1, column2, mode=mode))
            
            method.__doc__ = doc_string
            return method
        
        # Add each method to the DataFrame class
        for name, doc in [
            ("gt", "Filter rows where column1 > column2"),
            ("lt", "Filter rows where column1 < column2"),
            ("gte", "Filter rows where column1 >= column2"),
            ("lte", "Filter rows where column1 <= column2"),
            ("eq", "Filter rows where column1 == column2"),
            ("ne", "Filter rows where column1 != column2"),
            ("contains", "Filter rows where column1 contains column2 (for string columns)"),
            ("not_contains", "Filter rows where column1 does not contain column2 (for string columns)"),
            ("is_in", "Filter rows where column1 is in column2 (where column2 is a sequence)"),
            ("not_in", "Filter rows where column1 is not in column2 (where column2 is a sequence)"),
        ]:
            setattr(pl.DataFrame, name, _make_method(name, doc))

    @staticmethod
    def _get_expr(column: Union[pl.Series, str, pl.Expr], df: Optional[pl.DataFrame] = None) -> pl.Expr:
        """
        Convert a column identifier to a Polars expression.
        
        Args:
            column: Column as string name, Polars Series, or Polars Expression
            df: Optional DataFrame to use for context
            
        Returns:
            Polars expression
        """
        if isinstance(column, pl.Expr):
            return column
        elif isinstance(column, str):
            return pl.col(column)
        elif isinstance(column, pl.Series):
            return pl.lit(column)
        else:
            return pl.lit(column)

    @staticmethod
    def _compare_columns(
        column1: Union[str, pl.Series, pl.Expr, Any],
        column2: Union[str, pl.Series, pl.Expr, Sequence[Any], Any],
        operation: str,
        mode: str = "any"
    ) -> pl.Expr:
        """
        Compare two columns using the specified operation.
        
        Args:
            column1: First column to compare (string name, Polars Series, or Expression)
            column2: Second column or value to compare against
            operation: Operation to perform ("gt", "lt", "eq", etc.)
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        # Convert column1 to a list if it's not already
        if not isinstance(column1, list):
            column1 = [column1]
        
        # Map operations to their corresponding expression methods
        op_map = {
            "gt": lambda x, y: x > y,
            "lt": lambda x, y: x < y,
            "gte": lambda x, y: x >= y,
            "lte": lambda x, y: x <= y,
            "eq": lambda x, y: x == y,
            "ne": lambda x, y: x != y,
            "contains": lambda x, y: x.str.contains(y) if isinstance(y, str) else x.str.contains(str(y)),
            "not_contains": lambda x, y: ~x.str.contains(y) if isinstance(y, str) else ~x.str.contains(str(y)),
            "is_in": lambda x, y: x.is_in(y) if isinstance(y, (list, tuple, set)) else x == y,
            "not_in": lambda x, y: ~x.is_in(y) if isinstance(y, (list, tuple, set)) else x != y,
        }
        
        # Apply the operation to each column
        expressions = []
        for col in column1:
            col_expr = ColumnComparator._get_expr(col)
            
            # If column2 is a string, Series, or Expression, use it as-is
            if isinstance(column2, (str, pl.Series, pl.Expr)):
                col2_expr = ColumnComparator._get_expr(column2)
                expressions.append(op_map[operation](col_expr, col2_expr))
            else:
                # Otherwise, use it as a literal value
                expressions.append(op_map[operation](col_expr, column2))
        
        # Combine expressions based on mode
        if len(expressions) == 1:
            return expressions[0]
        
        if mode.lower() == "all":
            return pl.all_horizontal(expressions)
        return pl.any_horizontal(expressions)

    # Comparison methods
    @staticmethod
    def gt(column1, column2, mode="any"):
        """
        Check if column1 is greater than column2.
        
        Args:
            column1: First column(s) to compare (string name, Polars Series, or Expression)
            column2: Second column or value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        return ColumnComparator._compare_columns(column1, column2, "gt", mode)
    
    @staticmethod
    def lt(column1, column2, mode="any"):
        """
        Check if column1 is less than column2.
        
        Args:
            column1: First column(s) to compare (string name, Polars Series, or Expression)
            column2: Second column or value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        return ColumnComparator._compare_columns(column1, column2, "lt", mode)
    
    @staticmethod
    def gte(column1, column2, mode="any"):
        """
        Check if column1 is greater than or equal to column2.
        
        Args:
            column1: First column(s) to compare (string name, Polars Series, or Expression)
            column2: Second column or value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        return ColumnComparator._compare_columns(column1, column2, "gte", mode)
    
    @staticmethod
    def lte(column1, column2, mode="any"):
        """
        Check if column1 is less than or equal to column2.
        
        Args:
            column1: First column(s) to compare (string name, Polars Series, or Expression)
            column2: Second column or value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        return ColumnComparator._compare_columns(column1, column2, "lte", mode)
    
    @staticmethod
    def eq(column1, column2, mode="any"):
        """
        Check if column1 is equal to column2.
        
        Args:
            column1: First column(s) to compare (string name, Polars Series, or Expression)
            column2: Second column or value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        return ColumnComparator._compare_columns(column1, column2, "eq", mode)
    
    @staticmethod
    def ne(column1, column2, mode="any"):
        """
        Check if column1 is not equal to column2.
        
        Args:
            column1: First column(s) to compare (string name, Polars Series, or Expression)
            column2: Second column or value to compare against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        return ColumnComparator._compare_columns(column1, column2, "ne", mode)
    
    @staticmethod
    def contains(column1, pattern, mode="any"):
        """
        Check if string column(s) contain(s) a specific pattern.
        
        Args:
            column1: Column(s) to check (string name, Polars Series, or Expression)
            pattern: Pattern to search for
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        return ColumnComparator._compare_columns(column1, pattern, "contains", mode)
    
    @staticmethod
    def not_contains(column1, pattern, mode="any"):
        """
        Check if string column(s) do(es) not contain a specific pattern.
        
        Args:
            column1: Column(s) to check (string name, Polars Series, or Expression)
            pattern: Pattern to search for
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        return ColumnComparator._compare_columns(column1, pattern, "not_contains", mode)
    
    @staticmethod
    def is_in(column1, values, mode="any"):
        """
        Check if column1 value(s) are in a sequence of values.
        
        Args:
            column1: Column(s) to check (string name, Polars Series, or Expression)
            values: Sequence of values to check against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        return ColumnComparator._compare_columns(column1, values, "is_in", mode)
    
    @staticmethod
    def not_in(column1, values, mode="any"):
        """
        Check if column1 value(s) are not in a sequence of values.
        
        Args:
            column1: Column(s) to check (string name, Polars Series, or Expression)
            values: Sequence of values to check against
            mode: For multiple columns, 'any' or 'all' (default: 'any')
            
        Returns:
            Polars expression
        """
        return ColumnComparator._compare_columns(column1, values, "not_in", mode)

    # Longer aliases for better readability
    @staticmethod
    def greater_than(column1, column2, mode="any"):
        """Alias for gt"""
        return ColumnComparator.gt(column1, column2, mode)
    
    @staticmethod
    def less_than(column1, column2, mode="any"):
        """Alias for lt"""
        return ColumnComparator.lt(column1, column2, mode)
    
    @staticmethod
    def greater_than_equal_to(column1, column2, mode="any"):
        """Alias for gte"""
        return ColumnComparator.gte(column1, column2, mode)
    
    @staticmethod
    def less_than_equal_to(column1, column2, mode="any"):
        """Alias for lte"""
        return ColumnComparator.lte(column1, column2, mode)
    
    @staticmethod
    def equals(column1, column2, mode="any"):
        """Alias for eq"""
        return ColumnComparator.eq(column1, column2, mode)
    
    @staticmethod
    def not_equals(column1, column2, mode="any"):
        """Alias for ne"""
        return ColumnComparator.ne(column1, column2, mode)


# Install the methods to the DataFrame class
ColumnComparator._install_methods()


# Example usage
def example():
    # Create a sample DataFrame
    df = pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": ["apple", "banana", "cherry", "date", "elderberry"]
    })

    # Example 1: Using the new syntax to filter rows where A > 3
    filtered_df = df.gt("A", 3)
    print("Rows where A > 3:")
    print(filtered_df)

    # Example 2: Filter rows where A > B (column to column comparison)
    filtered_df = df.gt("A", "B")
    print("\nRows where A > B:")
    print(filtered_df)

    # Example 3: Filter rows where C contains 'a'
    filtered_df = df.contains("C", "a")
    print("\nRows where C contains 'a':")
    print(filtered_df)

    # Example 4: Filter rows where either A > 3 or B > 30
    filtered_df = df.gt(["A", "B"], [3, 30])  # default mode is "any"
    print("\nRows where either A > 3 or B > 30:")
    print(filtered_df)

    # Example 5: Filter rows where B is in [20, 40]
    filtered_df = df.is_in("B", [20, 40])
    print("\nRows where B is in [20, 40]:")
    print(filtered_df)

    # Example 6: Complex filtering with chaining
    filtered_df = df.gt("A", 2).lt("B", 40)
    print("\nRows where A > 2 AND B < 40:")
    print(filtered_df)

    # Example 7: Comparing a column with another column
    filtered_df = df.filter(ColumnComparator.gt("A", df["B"] / 10))
    print("\nRows where A > B/10:")
    print(filtered_df)

    # Example 8: Using expressions
    filtered_df = df.filter(ColumnComparator.gt(pl.col("A") + 1, "B" / 10))
    print("\nRows where (A + 1) > (B/10):")
    print(filtered_df)


if __name__ == "__main__":
    example()