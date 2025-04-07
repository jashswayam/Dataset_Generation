import polars as pl
from typing import List, Union, Dict, Any

class Aggregators:
    @staticmethod
    def get_aggregator(column: str, func: str, alias_name: str = None) -> Union[pl.Expr, None]:
        """
        Get the appropriate Polars aggregator expression based on function name
        
        Parameters:
        - column: Column to aggregate
        - func: Aggregation function name (mean, std, min, max, etc.)
        - alias_name: Optional alias for the result column
        
        Returns:
        - Polars expression for aggregation
        """
        if not alias_name:
            alias_name = f"{func}_{column}"
            
        if func.lower() == "mean":
            return pl.col(column).mean().alias(alias_name)
        elif func.lower() == "std":
            return pl.col(column).std().alias(alias_name)
        elif func.lower() == "min":
            return pl.col(column).min().alias(alias_name)
        elif func.lower() == "max":
            return pl.col(column).max().alias(alias_name)
        elif func.lower() == "sum":
            return pl.col(column).sum().alias(alias_name)
        elif func.lower() == "count":
            return pl.col(column).count().alias(alias_name)
        elif func.lower() == "median":
            return pl.col(column).median().alias(alias_name)
        elif func.lower() == "var":
            return pl.col(column).var().alias(alias_name)
        elif func.lower() == "first":
            return pl.col(column).first().alias(alias_name)
        elif func.lower() == "last":
            return pl.col(column).last().alias(alias_name)
        else:
            print(f"Warning: Unknown aggregation function '{func}'")
            return None

class DFoperations:
    @staticmethod
    def df_group_by(dataset_df, group_key, group_col, columns=None, functions=None):
        """
        Group a dataframe by keys and apply aggregation functions to columns
        
        Parameters:
        - dataset_df: Input dataframe
        - group_key: Keys to group by
        - group_col: Column to apply aggregation functions to
        - columns: List of columns for aggregation (defaults to group_col if None)
        - functions: List of aggregation functions to apply
        
        Returns:
        - Aggregated dataframe
        """
        # Ensure group_key is a list
        if isinstance(group_key, str):
            group_key = [group_key]
            
        # If columns not provided, use group_col
        if columns is None:
            columns = [group_col]
            
        # If functions is a string, convert to list
        if isinstance(functions, str):
            functions = [func.strip() for func in functions.split(",")]
        elif not isinstance(functions, list):
            functions = [functions]
            
        # Prepare aggregation expressions
        agg_exprs = []
        
        # For simple cases with one function and one column
        if len(functions) == 1 and len(columns) == 1:
            func = functions[0]
            col = columns[0]
            # Create an alias that will match the expected format in derived values
            alias_name = func
            agg_expr = Aggregators.get_aggregator(col, func, alias_name)
            if agg_expr is not None:
                agg_exprs.append(agg_expr)
        else:
            # For more complex cases with multiple functions or columns
            for func, col in zip(functions, columns):
                alias_name = f"{col}_{func}"
                agg_expr = Aggregators.get_aggregator(col, func, alias_name)
                if agg_expr is not None:
                    agg_exprs.append(agg_expr)
        
        # Apply groupby and aggregation
        if agg_exprs:
            dataset_df = dataset_df.group_by(group_key).agg(agg_exprs)
            
        return dataset_df
    
    @staticmethod
    def df_filters(dataset_df, filters):
        """
        Apply filters to a dataframe
        
        Parameters:
        - dataset_df: Input dataframe
        - filters: List of filter dictionaries with Column, Operator, and Value
        
        Returns:
        - Filtered dataframe
        """
        for flt in filters:
            column, operator, value = flt.get("Column"), flt.get("Operator"), flt.get("Value")
            
            if column and operator and value:
                # Handle different operator types
                if operator.lower() == "eq":
                    dataset_df = dataset_df.filter(pl.col(column) == value)
                elif operator.lower() == "neq":
                    dataset_df = dataset_df.filter(pl.col(column) != value)
                elif operator.lower() == "gt":
                    dataset_df = dataset_df.filter(pl.col(column) > value)
                elif operator.lower() == "lt":
                    dataset_df = dataset_df.filter(pl.col(column) < value)
                elif operator.lower() == "gte":
                    dataset_df = dataset_df.filter(pl.col(column) >= value)
                elif operator.lower() == "lte":
                    dataset_df = dataset_df.filter(pl.col(column) <= value)
                elif operator.lower() == "between":
                    # Assume value is in format "(lower, upper)" or similar
                    try:
                        lower, upper = eval(value)
                        dataset_df = dataset_df.filter((pl.col(column) >= lower) & (pl.col(column) <= upper))
                    except Exception as e:
                        print(f"Error parsing 'between' values {value}: {e}")
                elif operator.lower() == "in":
                    try:
                        values = eval(value)
                        dataset_df = dataset_df.filter(pl.col(column).is_in(values))
                    except Exception as e:
                        print(f"Error parsing 'in' values {value}: {e}")
                elif operator.lower() == "like":
                    dataset_df = dataset_df.filter(pl.col(column).str.contains(value))
                else:
                    print(f"Warning: Unsupported operator '{operator}'")
        
        return dataset_df