import dask.dataframe as dd
import numpy as np
import psutil
import time
import gc
import numba
from numba import jit, prange
from dask.diagnostics import ProgressBar
from dask import delayed

def get_memory_usage():
    """Returns the current memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

# Optimized Numba functions
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
    """Returns an array of Yes/No based on amount threshold."""
    result = np.empty(len(amounts), dtype='<U3')  # Preallocate string array
    for i in range(len(amounts)):
        result[i] = 'Yes' if amounts[i] > threshold else 'No'
    return result

# Optimized group-by aggregation using Dask + Numba
def fast_group_aggregate(df, group_cols, agg_cols):
    """
    Perform fast group-by aggregation using Numba with Dask.
    """
    df_dict = {col: df[col].values for col in df.columns}
    group_values = df[group_cols].drop_duplicates().compute()
    results = []

    for _, group_row in group_values.iterrows():
        mask = np.ones(len(df), dtype=bool)
        for col in group_cols:
            mask &= (df_dict[col] == group_row[col])

        if not np.any(mask):
            continue

        result_row = {col: group_row[col] for col in group_cols}

        for col in agg_cols:
            group_data = df_dict[col][mask]
            if len(group_data) == 0:
                continue

            if col == 'amount':
                result_row[f'{col}_sum'] = numba_sum(group_data)
                result_row[f'{col}_mean'] = numba_mean(group_data)
                result_row[f'{col}_count'] = len(group_data)
            elif col == 'balance':
                result_row[f'{col}_mean'] = numba_mean(group_data)
            elif col in ['account_id', 'merchant_id']:
                result_row[f'{col}_unique'] = numba_count_unique(group_data)

        results.append(result_row)

    return dd.from_pandas(pd.DataFrame(results), npartitions=1)

def perform_operations(input_dir="bank_data_joins"):
    """Main function for Dask + Numba processing."""
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    print("\nLoading data using Dask...")
    accounts_df = dd.read_parquet(f"{input_dir}/accounts.parquet", columns=['account_id', 'balance', 'status'])
    merchants_df = dd.read_parquet(f"{input_dir}/merchants.parquet", columns=['merchant_id', 'category'])
    transactions_df = dd.read_parquet(f"{input_dir}/transactions.parquet", columns=['transaction_id', 'account_id', 'merchant_id', 'amount'])

    print("\nFiltering data...")
    accounts_filtered = accounts_df[accounts_df['status'] == 'Active']

    print("\nProcessing transactions in parallel...")

    @delayed
    def process_chunk(txn_part):
        """Processes a chunk of transactions."""
        txn_pd = txn_part.compute() if isinstance(txn_part, dd.DataFrame) else txn_part  
        
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
    combined_results = dd.concat(chunk_results)

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