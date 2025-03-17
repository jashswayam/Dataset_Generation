import dask.dataframe as dd
import numpy as np
import psutil
import time
import numba
from numba import jit
from dask.diagnostics import ProgressBar
from dask import delayed

def get_memory_usage():
    """Returns the current memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

# Numba optimized functions
@jit(nopython=True, parallel=True)
def numba_sum(arr):
    return np.sum(arr)

@jit(nopython=True, parallel=True)
def numba_mean(arr):
    return np.mean(arr)

@jit(nopython=True)
def numba_count_unique(arr):
    return len(set(arr))

@jit(nopython=True)
def numba_filter_high_value(amounts, threshold=500):
    """Returns a NumPy array of Yes/No strings based on amount threshold."""
    result = np.empty(len(amounts), dtype='<U3')
    for i in range(len(amounts)):
        result[i] = 'Yes' if amounts[i] > threshold else 'No'
    return result

# Fast group-by aggregation
def fast_group_aggregate(df, group_cols, agg_cols):
    """Perform fast group-by aggregation using Numba & Dask."""
    df_grouped = df.groupby(group_cols).agg(
        amount_sum=('amount', 'sum'),
        amount_mean=('amount', 'mean'),
        amount_count=('amount', 'count'),
        balance_mean=('balance', 'mean'),
        account_id_unique=('account_id', 'nunique'),
        merchant_id_unique=('merchant_id', 'nunique')
    ).reset_index()
    
    return df_grouped

def perform_operations(input_dir="bank_data_joins"):
    """Main function to process data using Dask & Numba."""
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    print("\nLoading data using Dask...")
    accounts_df = dd.read_parquet(f"{input_dir}/accounts.parquet", columns=['account_id', 'balance', 'status'])
    merchants_df = dd.read_parquet(f"{input_dir}/merchants.parquet", columns=['merchant_id', 'category'])
    transactions_df = dd.read_parquet(f"{input_dir}/transactions.parquet", columns=['transaction_id', 'account_id', 'merchant_id', 'amount'])

    print("\nFiltering active accounts...")
    accounts_filtered = accounts_df[accounts_df['status'] == 'Active']

    print("\nProcessing transactions in parallel...")

    @delayed
    def process_chunk(txn_part):
        """Processes a chunk of transactions."""
        txn_pd = txn_part.compute()
        
        merged = txn_pd.merge(merchants_df.compute(), on='merchant_id', how='inner')
        merged = merged.merge(accounts_filtered.compute(), on='account_id', how='inner')

        merged['high_value_transaction'] = numba_filter_high_value(merged['amount'].values)

        return fast_group_aggregate(
            merged,
            group_cols=['category', 'high_value_transaction'],
            agg_cols=['amount', 'balance', 'account_id', 'merchant_id']
        )

    chunk_results = [process_chunk(partition) for partition in transactions_df.to_delayed()[:10]]

    print("\nCombining chunk results...")
    combined_results = dd.from_delayed(chunk_results)

    print("\nComputing final results...")
    with ProgressBar():
        final_result = combined_results.compute()

    print("\nSample output:")
    print(final_result.head())

    final_memory = get_memory_usage()
    print(f"\nMemory after processing: {final_memory:.2f} MB")

    final_result.to_parquet(f"{input_dir}/grouped_data_numba.parquet", compression='snappy')
    final_result.to_csv(f"{input_dir}/grouped_data_numba.csv")

    return final_result

if __name__ == "__main__":
    perform_operations()