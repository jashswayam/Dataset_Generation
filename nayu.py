
import polars as pl
from typing import Union

def any_pair_low_risk(expr1: pl.Expr, expr2: Union[str, pl.Expr]) -> pl.Expr:
    """
    Returns True if ANY pair in expr1 has BOTH currencies in expr2 (low-risk set).
    - expr1: pl.Expr like 'USD/INR,GBP/AED'
    - expr2: str or pl.Expr like 'USD,INR'
    """
    
    def check_any_pair(pair_str: str, low_risk_str: str) -> bool:
        low_risk_set = set(low_risk_str.split(","))
        pairs = pair_str.split(",")
        for pair in pairs:
            try:
                c1, c2 = pair.split("/")
                if c1 in low_risk_set and c2 in low_risk_set:
                    return True
            except ValueError:
                continue  # skip invalid pairs
        return False

    if isinstance(expr2, str):
        return expr1.map_elements(
            lambda pair_str: check_any_pair(pair_str, expr2),
            return_dtype=pl.Boolean
        )

    elif isinstance(expr2, pl.Expr):
        return pl.struct([expr1.alias("pair_str"), expr2.alias("low_risk_str")]).map_elements(
            lambda row: check_any_pair(row["pair_str"], row["low_risk_str"]),
            return_dtype=pl.Boolean
        )

    else:
        raise TypeError("expr2 must be a str or pl.Expr")

df = pl.DataFrame({
    "pairs": ["USD/GBP,EUR/USD", "USD/INR,GBP/AED", "USD/EUR"],
    "allowed": ["USD,EUR,GBP", "USD,INR", "USD,EUR"]
})

df = df.with_columns(
    all_pairs_low_risk(pl.col("pairs"), pl.col("allowed")).alias("low_risk_check")
)
print(df)