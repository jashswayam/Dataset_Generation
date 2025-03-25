from __future__ import annotations
from typing import Union, Any, Callable
import polars as pl
import pandas as pd
import operator

class ColumnComparator:
    # Comparison operators directly in the class
    eq = operator.eq
    ne = operator.ne
    gt = operator.gt
    ge = operator.ge
    lt = operator.lt
    le = operator.le

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
                column2: Union[pl.Series, pl.Expr, pd.Series, str, pl.LazyFrame]) -> pl.Expr:
        """
        Checks if all elements in column2 are contained in column1.
        
        Both columns contain comma-separated strings like "A,B,C,D".

        Args:
            column1: First column to check against
            column2: Column or string to check for containment
        
        Returns:
            Polars expression where True indicates all items in column2 are present in column1
        
        Raises:
            TypeError: For unsupported types
        """
        # If column2 is a string, create a simple subset check
        if isinstance(column2, str):
            column2_set = ColumnComparator._to_set(column2)
            return pl.col(column1).map(
                lambda x: column2_set.issubset(ColumnComparator._to_set(x))
            )
        
        # If column2 is an expression or column name
        elif isinstance(column2, (pl.Expr, str)):
            # Convert column name to expression if needed
            col2_expr = column2 if isinstance(column2, pl.Expr) else pl.col(column2)
            
            # Create an expression that checks subset containment
            return pl.struct([pl.col(column1), col2_expr]).map(
                lambda x: ColumnComparator._to_set(x[1]).issubset(
                    ColumnComparator._to_set(x[0])
                )
            )
        
        # Handle series comparison
        elif isinstance(column2, (pl.Series, pd.Series)):
            # Create an expression for subset comparison
            return pl.col(column1).map_batches(
                lambda x: [
                    ColumnComparator._to_set(val2).issubset(ColumnComparator._to_set(val1)) 
                    for val1, val2 in zip(x, column2)
                ]
            )
        
        raise TypeError("Unsupported types for list_in comparison")

    @classmethod
    def create_operator(cls, op_func: Callable):
        """
        Create a custom operator function for comparisons.
        
        Args:
            op_func: Comparison function to apply
        
        Returns:
            A function that can be used as an operator
        """
        def _operator(column1, column2):
            # If column2 is a simple value or string
            if isinstance(column2, (str, int, float)):
                return pl.col(column1).map(lambda x: op_func(x, column2))
            
            # If column2 is an expression or column name
            elif isinstance(column2, (pl.Expr, str)):
                # Convert column name to expression if needed
                col2_expr = column2 if isinstance(column2, pl.Expr) else pl.col(column2)
                
                # Create an expression that applies the comparison
                return pl.struct([pl.col(column1), col2_expr]).map(
                    lambda x: op_func(x[0], x[1])
                )
            
            raise TypeError("Unsupported types for operator comparison")
        
        return _operator

    @staticmethod
    def __call__(column1: Union[pl.Series, pl.Expr, pd.Series, pl.LazyFrame], 
                 column2: Union[pl.Series, pl.Expr, pd.Series, str, pl.LazyFrame]) -> pl.Expr:
        """
        Allows using the method as an operator-like function.
        
        Args:
            column1: First column to check against
            column2: Column or string to check for containment
        
        Returns:
            Polars expression for the comparison
        """
        return ColumnComparator.list_in(column1, column2)

# Instantiate the comparator
list_in = ColumnComparator()

# Demonstration function
def demonstrate_comparators():
    # Create a sample LazyFrame
    lf = pl.LazyFrame({
        'source_currencies': ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
        'target_currencies': ['AED,INR', 'EUR', 'AED,YEN'],
        'amount': [100, 200, 300]
    })
    
    # Case 1: List-in comparison
    result1 = lf.filter(
        list_in(pl.col("source_currencies"), pl.col("target_currencies"))
    ).collect()
    print("List-in comparison:", result1)
    
    # Case 2: Greater than comparison using class operator
    result2 = lf.filter(
        ColumnComparator.gt(pl.col("amount"), 150)
    ).collect()
    print("\nGreater than 150:", result2)
    
    # Case 3: Complex filtering
    result3 = lf.filter(
        (list_in(pl.col("source_currencies"), "INR,AED")) & 
        (ColumnComparator.gt(pl.col("amount"), 100))
    ).collect()
    print("\nComplex filtering:", result3)
    
    # Case 4: Equality comparison
    result4 = lf.filter(
        ColumnComparator.eq(pl.col("amount"), 200)
    ).collect()
    print("\nEqual to 200:", result4)

# Example of custom comparison and operators
def custom_comparison():
    # Create a custom comparison function
    def between(column1, lower, upper):
        return ColumnComparator.create_operator(
            lambda x, _: lower <= x <= upper
        )(column1, None)
    
    # Create a sample LazyFrame
    lf = pl.LazyFrame({
        'amount': [100, 200, 300, 400, 500]
    })
    
    # Filter amounts between 150 and 350
    result = lf.filter(
        between(pl.col("amount"), 150, 350)
    ).collect()
    print("Amounts between 150 and 350:", result)

# Uncomment to run demonstrations
# demonstrate_comparators()
# custom_comparison()