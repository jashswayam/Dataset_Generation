import polars as pl
import pandas as pd
from typing import Union, Optional, Any

class ListInComparator:
    @staticmethod
    def _to_set(x: Optional[str]) -> set:
        """
        Convert comma-separated string to a set of stripped items.
        
        Args:
            x (str, optional): Comma-separated string
        
        Returns:
            set: Set of stripped items
        """
        if not isinstance(x, str):
            return set()
        return {item.strip() for item in x.split(',')}

    @staticmethod
    def list_in(
        column1: Union[pl.Series, pl.Expr, pd.Series, pl.LazyFrame, pl.DataFrame], 
        column2: Union[str, pl.Series, pl.Expr, pd.Series, list]
    ) -> Union[pl.Series, pl.Expr, pd.Series, pl.LazyFrame]:
        """
        Checks if all elements in column2 are contained in column1.
        
        Supports multiple input types and contexts:
        - Polars eager Series/DataFrame
        - Polars lazy DataFrame
        - Pandas Series
        - Scalar string or list
        
        Args:
            column1: First column of comma-separated strings
            column2: Second column of comma-separated strings or a single string/list
        
        Returns:
            Boolean series/expression indicating if all items in column2 are present in column1
        """
        # Normalize column2 to a set if it's a string or list
        if isinstance(column2, (str, list)):
            # Convert to set, handling both string and list inputs
            column2_set = ListInComparator._to_set(column2 if isinstance(column2, str) else ','.join(column2))

            # Polars Lazy DataFrame handling
            if isinstance(column1, pl.LazyFrame):
                return column1.with_columns(
                    pl.col(column1.columns[0]).apply(
                        lambda x: column2_set.issubset(ListInComparator._to_set(x))
                    ).alias('list_in_result')
                )

            # Polars Eager Series/DataFrame/Expr handling
            if isinstance(column1, (pl.Series, pl.Expr, pl.DataFrame)):
                # If it's a DataFrame, assume first column
                if isinstance(column1, pl.DataFrame):
                    column1 = column1.get_column(column1.columns[0])
                
                # For Polars expressions
                if isinstance(column1, pl.Expr):
                    return column1.apply(lambda x: column2_set.issubset(ListInComparator._to_set(x)))
                
                # For Polars series
                return column1.apply(lambda x: column2_set.issubset(ListInComparator._to_set(x)))

            # Pandas Series handling
            if isinstance(column1, pd.Series):
                return column1.apply(lambda x: column2_set.issubset(ListInComparator._to_set(x)))

        # Series to Series comparison
        # Polars Lazy DataFrame handling
        if isinstance(column1, pl.LazyFrame):
            return column1.with_columns(
                pl.struct([pl.col(column1.columns[0]), pl.col(column1.columns[1])])
                .apply(lambda row: ListInComparator._to_set(row[1]).issubset(
                    ListInComparator._to_set(row[0])
                ))
                .alias('list_in_result')
            )

        # Polars Eager Series comparison
        if isinstance(column1, (pl.Series, pl.Expr)) and isinstance(column2, (pl.Series, pl.Expr)):
            return pl.Series([
                ListInComparator._to_set(col2).issubset(ListInComparator._to_set(col1))
                for col1, col2 in zip(column1, column2)
            ])

        # Pandas Series comparison
        if isinstance(column1, pd.Series) and isinstance(column2, pd.Series):
            return pd.Series([
                ListInComparator._to_set(col2).issubset(ListInComparator._to_set(col1))
                for col1, col2 in zip(column1, column2)
            ])

        raise TypeError("Unsupported types for list_in comparison")

    # Example usage and demonstration
    @staticmethod
    def demonstrate_usage():
        # Polars Eager Mode Examples
        print("Polars Eager Mode Examples:")
        # Case 1: Series to Series
        A_pl = pl.Series('A', ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'])
        B_pl = pl.Series('B', ['AED,INR', 'EUR', 'AED,YEN'])
        print("\nEager Series to Series:")
        print("A:", A_pl)
        print("B:", B_pl)
        print("Result:", ListInComparator.list_in(A_pl, B_pl))

        # Case 2: Series to Scalar
        A_pl_scalar = pl.Series('A', ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'])
        B_pl_scalar = " INR,AED"
        print("\nEager Series to Scalar:")
        print("A:", A_pl_scalar)
        print("B:", B_pl_scalar)
        print("Result:", ListInComparator.list_in(A_pl_scalar, B_pl_scalar))

        # Polars Lazy Mode Examples
        print("\nPolars Lazy Mode Examples:")
        # Lazy DataFrame Example
        df_lazy = pl.LazyFrame({
            'currency': ['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'],
            'blacklist': ['AED,INR', 'EUR', 'AED,YEN']
        })
        
        # Lazy filter example
        print("\nLazy Filter Example:")
        result_lazy = (
            df_lazy
            .filter(
                ListInComparator.list_in(pl.col('currency'), pl.col('blacklist'))
            )
            .collect()
        )
        print("Filtered Result:", result_lazy)

        # Pandas Examples
        print("\nPandas Examples:")
        # Case 1: Series to Series
        A_pd = pd.Series(['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'])
        B_pd = pd.Series(['AED,INR', 'EUR', 'AED,YEN'])
        print("\nPandas Series to Series:")
        print("A:", A_pd)
        print("B:", B_pd)
        print("Result:", ListInComparator.list_in(A_pd, B_pd))

        # Case 2: Series to Scalar
        A_pd_scalar = pd.Series(['INR,AED,EUR', 'INR,EUR', 'USD,INR,EUR'])
        B_pd_scalar = " INR,AED"
        print("\nPandas Series to Scalar:")
        print("A:", A_pd_scalar)
        print("B:", B_pd_scalar)
        print("Result:", ListInComparator.list_in(A_pd_scalar, B_pd_scalar))

# Uncomment to run demonstration
# ListInComparator.demonstrate_usage()