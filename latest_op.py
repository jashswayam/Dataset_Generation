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

    @classmethod
    def _ensure_expr(cls, column: Union[str, pl.Expr]) -> pl.Expr:
        """
        Ensure the input is a Polars expression.
        
        Args:
            column: Column name or existing expression
        
        Returns:
            Polars expression
        """
        return column if isinstance(column, pl.Expr) else pl.col(column)

    @classmethod
    def list_in(cls, column1: Union[str, pl.Expr], 
                column2: Union[str, pl.Expr, list[str]]) -> pl.Expr:
        """
        Checks if all elements in column2 are contained in column1.
        
        Both columns contain comma-separated strings like "A,B,C,D".

        Args:
            column1: First column to check against
            column2: Column or string to check for containment
        
        Returns:
            Polars expression where True indicates all items in column2 are present in column1
        """
        # Ensure both inputs are expressions
        col1_expr = cls._ensure_expr(column1)
        
        # If column2 is a simple string or list of strings
        if isinstance(column2, (str, list)):
            # Convert to set of strings
            column2_set = (cls._to_set(column2) 
                           if isinstance(column2, str) 
                           else set(str(x).strip() for x in column2))
            
            return col1_expr.map(
                lambda x: column2_set.issubset(cls._to_set(x))
            )
        
        # If column2 is an expression
        col2_expr = cls._ensure_expr(column2)
        
        # Create an expression that checks subset containment
        return pl.struct([col1_expr, col2_expr]).map(
            lambda x: cls._to_set(x[1]).issubset(cls._to_set(x[0]))
        )

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
            # Ensure column1 is an expression
            col1_expr = cls._ensure_expr(column1)
            
            # If column2 is a simple value
            if isinstance(column2, (str, int, float, bool)):
                return col1_expr.map(lambda x: op_func(x, column2))
            
            # If column2 is an expression or column name
            col2_expr = cls._ensure_expr(column2)
            
            # Create an expression that applies the comparison
            return pl.struct([col1_expr, col2_expr]).map(
                lambda x: op_func(x[0], x[1])
            )
        
        return _operator

    @staticmethod
    def __call__(column1: Union[str, pl.Expr], 
                 column2: Union[str, pl.Expr, list[str]]) -> pl.Expr:
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
    # Create a sample DataFrame
    df = pl.DataFrame({
        'source_currencies': ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
        'target_currencies': ['AED,INR', 'EUR', 'AED,YEN'],
        'amount': [100, 200, 300]
    })
    
    # Case 1: List-in comparison with column names
    result1 = df.filter(
        list_in('source_currencies', 'target_currencies')
    )
    print("List-in comparison:", result1)
    
    # Case 2: List-in comparison with string
    result2 = df.filter(
        list_in('source_currencies', 'INR,AED')
    )
    print("\nList-in with string:", result2)
    
    # Case 3: Greater than comparison using class operator
    result3 = df.filter(
        ColumnComparator.gt('amount', 150)
    )
    print("\nGreater than 150:", result3)
    
    # Case 4: Complex filtering
    result4 = df.filter(
        (list_in('source_currencies', 'INR,AED')) & 
        (ColumnComparator.gt('amount', 100))
    )
    print("\nComplex filtering:", result4)
    
    # Case 5: Equality comparison
    result5 = df.filter(
        ColumnComparator.eq('amount', 200)
    )
    print("\nEqual to 200:", result5)

# Uncomment to run demonstration
demonstrate_comparators()