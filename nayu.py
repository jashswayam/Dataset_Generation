import polars as pl
from typing import Union

def all_pairs_low_risk(expr1: pl.Expr, expr2: Union[str, pl.Expr]) -> pl.Expr:
    """
    expr1: pl.Expr containing strings like 'USD/GBP,EUR/USD'
    expr2: either a comma-separated string like 'USD,EUR,GBP' or a pl.Expr with similar strings
    
    Returns: pl.Expr[Boolean] → True if all currencies in all pairs in expr1 are present in expr2
    """
    
    def check_all_pairs(pair_str: str, low_risk_str: str) -> bool:
        low_risk_set = set(low_risk_str.split(","))
        pairs = pair_str.split(",")
        for p in pairs:
            try:
                c1, c2 = p.split("/")
                if c1 not in low_risk_set or c2 not in low_risk_set:
                    return False
            except ValueError:
                return False  # malformed input like no '/' in pair
        return True

    if isinstance(expr2, str):
        # expr2 is static, pass it directly
        return expr1.map_elements(
            lambda pair_str: check_all_pairs(pair_str, expr2),
            return_dtype=pl.Boolean
        )

    elif isinstance(expr2, pl.Expr):
        # expr2 is a dynamic column, zip expr1 and expr2 together
        return pl.struct([expr1.alias("pair_str"), expr2.alias("low_risk_str")]).map_elements(
            lambda row: check_all_pairs(row["pair_str"], row["low_risk_str"]),
            return_dtype=pl.Boolean
        )

    else:
        raise TypeError("expr2 must be a str or a pl.Expr")


df = pl.DataFrame({
    "pairs": ["USD/GBP,EUR/USD", "USD/INR,GBP/AED", "USD/EUR"],
    "allowed": ["USD,EUR,GBP", "USD,INR", "USD,EUR"]
})

df = df.with_columns(
    all_pairs_low_risk(pl.col("pairs"), pl.col("allowed")).alias("low_risk_check")
)
print(df)