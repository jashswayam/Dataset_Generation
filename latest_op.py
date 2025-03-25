from typing import Union, Any
import polars as pl
import pandas as pd

class ColumnComparator:
    """
    A utility class for advanced column comparisons with support for Polars and Pandas.
    """
    @staticmethod
    def _validate_first_arg_is_valid(column: Any) -> None:
        """
        Validate the first argument is of a supported type.
        
        Args:
            column: Input column to validate
        
        Raises:
            TypeError: If the column is not a supported type
        """
        if not isinstance(column, (pl.Series, pl.Expr, pd.Series)):
            raise TypeError(f"Unsupported type for first argument: {type(column)}")

    @staticmethod
    def _validate_compatible_types(column1: Any, column2: Any) -> None:
        """
        Validate that the two columns are compatible for comparison.
        
        Args:
            column1: First column
            column2: Second column
        
        Raises:
            TypeError: If columns are incompatible
            ValueError: If series have different lengths
        """
        # Check types are supported
        if not isinstance(column1, (pl.Series, pl.Expr, pd.Series)) or \
           not isinstance(column2, (pl.Series, pl.Expr, pd.Series, str)):
            raise TypeError(f"Unsupported types for comparison: {type(column1)}, {type(column2)}")
        
        # Check length for series-to-series comparison
        if isinstance(column1, (pl.Series, pd.Series)) and \
           isinstance(column2, (pl.Series, pd.Series)) and \
           len(column1) != len(column2):
            raise ValueError("Series must have the same length for element-wise comparison")

    @staticmethod
    def _to_set(x: str) -> set:
        """
        Convert a comma-separated string to a set of stripped items.
        
        Args:
            x: Comma-separated string
        
        Returns:
            Set of stripped items
        """
        return set(item.strip() for item in str(x).split(',') if item.strip())

    @staticmethod
    def list_in(column1: Union[pl.Series, pl.Expr, pd.Series], 
                column2: Union[pl.Series, pl.Expr, pd.Series, str]) -> Union[pl.Series, pl.Expr, pd.Series]:
        """
        Checks if all elements in column2 are contained in column1.
        
        Both columns contain comma-separated strings like "A,B,C,D".
        
        Args:
            column1: First column to check against (source column)
            column2: Second column or string to check (items to find)
        
        Returns:
            Boolean series where True indicates all items in column2 are present in column1
        
        Examples:
            # Case 1: Series comparison
            A = ['INR,AED,EUR', 'INR,EUR','USD,INR,EUR']
            B = ['AED,INR', 'EUR', 'AED,YEN']
            # Output: [True, True, False]
            
            # Case 2: Series vs String
            A = ['INR,AED,EUR', 'INR,EUR','USD,INR,EUR']
            B = "INR,AED"
            # Output: [True, True, False]
        """
        # Validate first argument
        ColumnComparator._validate_first_arg_is_valid(column1)

        # Handle scalar case (string comparison against whole series)
        if isinstance(column2, str):
            column2_set = ColumnComparator._to_set(column2)

            # If using Polars
            if isinstance(column1, (pl.Series, pl.Expr)):
                return column1.apply(lambda x: column2_set.issubset(ColumnComparator._to_set(x)))
            # If using Pandas
            elif isinstance(column1, pd.Series):
                return column1.apply(lambda x: column2_set.issubset(ColumnComparator._to_set(x)))

        # Handle series to series comparison
        else:
            ColumnComparator._validate_compatible_types(column1, column2)

            # If using Polars
            if isinstance(column1, (pl.Series, pl.Expr)):
                return pl.Series([
                    ColumnComparator._to_set(col2).issubset(ColumnComparator._to_set(col1))
                    for col1, col2 in zip(column1, column2)
                ])
            # If using Pandas
            elif isinstance(column1, pd.Series):
                return pd.Series([
                    ColumnComparator._to_set(col2).issubset(ColumnComparator._to_set(col1))
                    for col1, col2 in zip(column1, column2)
                ])

        raise TypeError("Unsupported types for list_in comparison")

    # Operator-like method for easy chaining
    def __init__(self, column):
        """
        Initialize ColumnComparator with a column.
        
        Args:
            column: Input column (Polars or Pandas series)
        """
        self.column = column

    def __call__(self, other):
        """
        Enable easy chaining like: ColumnComparator(A).list_in(B)
        
        Args:
            other: Column or string to compare against
        
        Returns:
            Boolean series of comparison results
        """
        return ColumnComparator.list_in(self.column, other)