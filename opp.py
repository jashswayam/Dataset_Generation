from __future__ import annotations
from typing import Union, Any, Sequence
import polars as pl


class ColumnComparator:
    """
    Extension methods for Polars DataFrame to generate comparison expressions.
    These expressions can be used in `.filter()`, `.select()`, `.with_columns()`, etc.
    """

    @staticmethod
    def gt(column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 > column2."""
        return ColumnComparator._expr(column1) > ColumnComparator._expr(column2)

    @staticmethod
    def lt(column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 < column2."""
        return ColumnComparator._expr(column1) < ColumnComparator._expr(column2)

    @staticmethod
    def gte(column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 >= column2."""
        return ColumnComparator._expr(column1) >= ColumnComparator._expr(column2)

    @staticmethod
    def lte(column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 <= column2."""
        return ColumnComparator._expr(column1) <= ColumnComparator._expr(column2)

    @staticmethod
    def eq(column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 == column2."""
        return ColumnComparator._expr(column1) == ColumnComparator._expr(column2)

    @staticmethod
    def ne(column1: Union[str, pl.Expr, Any], column2: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Return an expression for column1 != column2."""
        return ColumnComparator._expr(column1) != ColumnComparator._expr(column2)

    @staticmethod
    def contains(column: Union[str, pl.Expr], pattern: str) -> pl.Expr:
        """Return an expression checking if a column contains a pattern."""
        return ColumnComparator._expr(column).str.contains(pattern)

    @staticmethod
    def not_contains(column: Union[str, pl.Expr], pattern: str) -> pl.Expr:
        """Return an expression checking if a column does not contain a pattern."""
        return ~ColumnComparator._expr(column).str.contains(pattern)

    @staticmethod
    def is_in(column: Union[str, pl.Expr], values: Sequence[Any]) -> pl.Expr:
        """Return an expression checking if column values are in a given sequence."""
        return ColumnComparator._expr(column).is_in(values)

    @staticmethod
    def not_in(column: Union[str, pl.Expr], values: Sequence[Any]) -> pl.Expr:
        """Return an expression checking if column values are NOT in a given sequence."""
        return ~ColumnComparator._expr(column).is_in(values)

    @staticmethod
    def _expr(column: Union[str, pl.Expr, Any]) -> pl.Expr:
        """Helper function to convert column names or values to a Polars expression."""
        if isinstance(column, str):
            return pl.col(column)
        elif isinstance(column, pl.Expr):
            return column
        else:
            return pl.lit(column)


# Example usage
def example():
    df = pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": ["apple", "banana", "cherry", "date", "elderberry"]
    })

    # Example 1: Using expressions in filtering
    print("\nRows where A > 3:")
    print(df.filter(ColumnComparator.gt("A", 3)))

    # Example 2: Using expressions in `select`
    print("\nSelected transformed column (A * 2 where A > 2):")
    print(df.select((pl.col("A") * 2).alias("A_times_2"), ColumnComparator.gt("A", 2).alias("A_gt_2")))

    # Example 3: Combining expressions
    print("\nRows where A > 2 AND B < 40:")
    print(df.filter(ColumnComparator.gt("A", 2) & ColumnComparator.lt("B", 40)))

    # Example 4: Using expressions in `with_columns`
    print("\nAdding a new column 'A_gt_B' based on expression:")
    print(df.with_columns(ColumnComparator.gt("A", "B").alias("A_gt_B")))

    # Example 5: String contains example
    print("\nRows where C contains 'a':")
    print(df.filter(ColumnComparator.contains("C", "a")))

    # Example 6: is_in example
    print("\nRows where B is in [20, 40]:")
    print(df.filter(ColumnComparator.is_in("B", [20, 40])))


if __name__ == "__main__":
    example()