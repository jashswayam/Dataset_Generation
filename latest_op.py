from __future__ import annotations
from typing import Union, Any
import polars as pl
import pandas as pd

class ColumnComparator:
    @staticmethod
    def _to_set(x: str) -> set[str]:
        """
        Convert a comma-separated string to a set of stripped items.
        
        Args:
            x: Comma-separated string
        
        Returns:
            Set of stripped items
        """
        return set(item.strip() for item in str(x).split(',') if item.strip())

    @staticmethod
    def list_in(column1: Union[pl.Series, pl.Expr, pd.Series, pl.LazyFrame], 
                column2: Union[pl.Series, pl.Expr, pd.Series, str, pl.LazyFrame, pl.col]) -> Union[pl.Series, pl.Expr, pd.Series, pl.LazyFrame]:
        """
        Checks if all elements in column2 are contained in column1.
        
        Both columns contain comma-separated strings like "A,B,C,D".

        Args:
            column1: First column to check against
            column2: Column or string to check for containment
        
        Returns:
            Boolean series/frame where True indicates all items in column2 are present in column1
        
        Raises:
            TypeError: For unsupported types
        """
        # Handle Polars LazyFrame cases
        if isinstance(column1, pl.LazyFrame):
            # If column2 is a column expression
            if isinstance(column2, pl.Expr):
                return column1.with_columns(
                    pl.col(column1.columns).apply(
                        lambda x, col2: ColumnComparator._to_set(col2).issubset(ColumnComparator._to_set(x)),
                        column2
                    )
                )
            
            # If column2 is a LazyFrame, we need to handle it differently
            if isinstance(column2, pl.LazyFrame):
                raise TypeError("Cannot compare two LazyFrames directly")
            
            # If column2 is a string, apply to all columns
            if isinstance(column2, str):
                column2_set = ColumnComparator._to_set(column2)
                return column1.with_columns(
                    pl.all().apply(lambda x: column2_set.issubset(ColumnComparator._to_set(x)))
                )
            
            # If column2 is a Series 
            return column1.with_columns(
                column1.select(column2).apply(
                    lambda x: column2.issubset(ColumnComparator._to_set(x))
                )
            )

        # Handle scalar case (string comparison against whole series)
        if isinstance(column2, str):
            column2_set = ColumnComparator._to_set(column2)

            # If using Polars
            if isinstance(column1, (pl.Series, pl.Expr)):
                return column1.apply(lambda x: column2_set.issubset(ColumnComparator._to_set(x)))
            
            # If using Pandas
            elif isinstance(column1, pd.Series):
                return column1.apply(lambda x: column2_set.issubset(ColumnComparator._to_set(x)))

        # Handle column expression case for Polars
        elif isinstance(column2, pl.Expr):
            # Create an expression that checks subset containment
            return column1.zip_with(column2, 
                lambda x, y: ColumnComparator._to_set(y).issubset(ColumnComparator._to_set(x))
            )

        # Handle series to series comparison (must be same length)
        else:
            # If using Polars
            if isinstance(column1, (pl.Series, pl.Expr)):
                return column1.zip_with(column2, 
                    lambda x, y: ColumnComparator._to_set(y).issubset(ColumnComparator._to_set(x))
                )
            
            # If using Pandas
            elif isinstance(column1, pd.Series):
                return pd.Series([
                    ColumnComparator._to_set(col2).issubset(ColumnComparator._to_set(col1))
                    for col1, col2 in zip(column1, column2)
                ])

        raise TypeError("Unsupported types for list_in comparison")

    @staticmethod
    def __call__(column1: Union[pl.Series, pl.Expr, pd.Series, pl.LazyFrame], 
                 column2: Union[pl.Series, pl.Expr, pd.Series, str, pl.LazyFrame, pl.col]) -> Union[pl.Series, pl.Expr, pd.Series, pl.LazyFrame]:
        """
        Allows using the method as an operator-like function.
        
        Args:
            column1: First column to check against
            column2: Column or string to check for containment
        
        Returns:
            Boolean series where True indicates all items in column2 are present in column1
        """
        return ColumnComparator.list_in(column1, column2)

# Instantiate the comparator
list_in = ColumnComparator()

# Demonstration function
def demonstrate_list_in():
    # Create a sample LazyFrame
    lf = pl.LazyFrame({
        'source_currencies': ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
        'target_currencies': ['AED,INR', 'EUR', 'AED,YEN']
    })
    
    # Case 1: Using pl.col() as second argument
    result1 = lf.filter(
        list_in(pl.col("source_currencies"), pl.col("target_currencies"))
    ).collect()
    print("Filtered using pl.col():", result1)
    
    # Case 2: Complex filtering with pl.col()
    result2 = lf.with_columns(
        list_in(pl.col("source_currencies"), pl.col("target_currencies")).alias("is_subset")
    ).collect()
    print("\nAdded check column:", result2)
    
    # Case 3: Mixing string and column expressions
    result3 = lf.filter(
        (list_in(pl.col("source_currencies"), "INR,AED")) & 
        (list_in(pl.col("source_currencies"), pl.col("target_currencies")))
    ).collect()
    print("\nComplex filtering:", result3)

# Standalone Polars DataFrame example
def standalone_example():
    # Create a DataFrame
    df = pl.DataFrame({
        'source_currencies': ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
        'target_currencies': ['AED,INR', 'EUR', 'AED,YEN']
    })
    
    # Filter using column comparison
    result = df.filter(
        list_in(pl.col("source_currencies"), pl.col("target_currencies"))
    )
    print("Filtered DataFrame:", result)

# Uncomment to run demonstrations
# demonstrate_list_in()
# standalone_example()