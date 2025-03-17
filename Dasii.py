import dask.dataframe as dd
import numpy as np
import pandas as pd
import psutil
import time
import gc
import numba
from numba import jit, prange
from dask.diagnostics import ProgressBar
from dask import delayed

def get_memory_usage():
    """Returns the current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Memory in MB

# Define Numba-accelerated functions for aggregation
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
    """Return array of Yes/No strings based on amount threshold"""
    result = np.empty(len(amounts), dtype='<U3')  # Preallocate string array
    for i in range(len(amounts)):
        result[i] = 'Yes' if amounts[i] > threshold else 'No'
    return result

# Custom aggregation function using Numba
def fast_group_aggregate(df, group_cols, agg_cols):
    """
    Perform fast group by aggregation using Numba-accelerated functions.
    """
    df_dict = {col: df[col].values for col in df.columns}
    group_values = df[group_cols].drop_duplicates()
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

    return pd.DataFrame(results)

def perform_operations(input_dir="bank_data_joins"):
    """Main function to process data efficiently with Dask and Numba."""
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Load data with Dask (lazy loading)
    print("\nLoading data using Dask...")
    accounts_df = dd.read_parquet(f"{input_dir}/accounts.parquet", engine='pyarrow', columns=['account_id', 'balance', 'status'])
    merchants_df = dd.read_parquet(f"{input_dir}/merchants.parquet", engine='pyarrow', columns=['merchant_id', 'category'])
    transactions_df = dd.read_parquet(f"{input_dir}/transactions.parquet", engine='pyarrow', columns=['transaction_id', 'account_id', 'merchant_id', 'amount'])

    loading_memory = get_memory_usage()
    print(f"Memory after loading data: {loading_memory:.2f} MB")

    print("\nFiltering and optimizing data processing...")

    # Filter active accounts
    accounts_filtered = accounts_df[accounts_df['status'] == 'Active']

    # Convert small dataframes to Pandas for faster joins
    if isinstance(accounts_filtered, dd.DataFrame):
        accounts_pd = accounts_filtered.compute()
    else:
        accounts_pd = accounts_filtered

    if isinstance(merchants_df, dd.DataFrame):
        merchants_pd = merchants_df.compute()
    else:
        merchants_pd = merchants_df

    # Processing transactions in chunks
    chunk_results = []

    @delayed
    def process_chunk(txn_part):
        txn_pd = txn_part.compute()
        
        # Merge transactions with merchants and accounts
        merged = txn_pd.merge(merchants_pd, on='merchant_id', how='inner')
        merged = merged.merge(accounts_pd, on='account_id', how='inner')

        # Apply high-value transaction filter
        amounts = merged['amount'].values
        merged['high_value_transaction'] = numba_filter_high_value(amounts)

        # Perform fast group-by
        result = fast_group_aggregate(
            merged,
            group_cols=['category', 'high_value_transaction'],
            agg_cols=['amount', 'balance', 'account_id', 'merchant_id']
        )
        
        return result

    print("Processing transactions in parallel...")
    for i, partition in enumerate(transactions_df.to_delayed()):
        if i >= 10:  # Limit partitions to avoid excessive memory use
            break
        chunk_results.append(process_chunk(partition))

    print("Combining chunk results...")
    combined_results = delayed(pd.concat)(chunk_results, ignore_index=True)

    # Final aggregation
    @delayed
    def final_aggregation(combined_df):
        return combined_df.groupby(['category', 'high_value_transaction']).agg({
            'amount_sum': 'sum',
            'amount_mean': 'mean',
            'amount_count': 'sum',
            'balance_mean': 'mean',
            'account_id_unique': 'sum',
            'merchant_id_unique': 'sum'
        }).reset_index()

    final_result_delayed = final_aggregation(combined_results)

    print("\nComputing final results with progress bar...")
    collection_start_time = time.time()
    with ProgressBar():
        result = final_result_delayed.compute()
    
    collection_time = time.time() - collection_start_time
    print(f"Final computation time: {collection_time:.4f} seconds.")

    print("\nSample output:")
    print(result.head())

    # Cleanup memory
    gc.collect()
    after_pipeline_memory = get_memory_usage()
    print(f"Memory after processing: {after_pipeline_memory:.2f} MB")

    # Save results
    result.to_parquet(f"{input_dir}/grouped_data_numba.parquet", compression='snappy')
    result.to_csv(f"{input_dir}/grouped_data_numba.csv")

    return result

if __name__ == "__main__":
    perform_operations()