import dask.dataframe as dd
import psutil
import time
import gc
import numba
import numpy as np
from distributed import Client

def get_memory_usage():
    """Returns the current memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert bytes to MB

@numba.njit
def mark_high_value_transaction(amount):
    """Numba-optimized function to classify high-value transactions."""
    return "Yes" if amount > 500 else "No"

def perform_operations(input_dir="bank_data_joins"):
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Load the generated data with lazy execution
    print("\nLazy loading data...")
    accounts_df = dd.read_parquet(f"{input_dir}/accounts.parquet")
    merchants_df = dd.read_parquet(f"{input_dir}/merchants.parquet")
    transactions_df = dd.read_parquet(f"{input_dir}/transactions.parquet")

    loading_memory = get_memory_usage()
    print(f"Memory after lazy loading: {loading_memory:.2f} MB")

    # ------ Merging Operation ------
    print("\nPerforming merging operation...")
    start_merge_time = time.time()

    accounts_transactions = accounts_df.merge(transactions_df, 
                                              left_on='account_id', 
                                              right_on='account_id', 
                                              how='inner')

    merged_df = accounts_transactions.merge(merchants_df, 
                                            left_on='merchant_id', 
                                            right_on='merchant_id', 
                                            how='inner')

    # Convert amount to NumPy for fast computation
    print("Applying optimized transaction classification...")
    merged_df['high_value_transaction'] = merged_df['amount'].map_partitions(
        lambda x: np.array([mark_high_value_transaction(a) for a in x.to_numpy()], dtype=str),
        meta=('amount', 'str')
    )

    after_merging_memory = get_memory_usage()
    print(f"Memory after merging: {after_merging_memory:.2f} MB")

    # ------ Group By Operation ------
    print("\nPerforming optimized group by operation...")
    start_groupby_time = time.time()

    # Optimize groupby by reducing partitions first
    partitions = min(merged_df.npartitions, 32)
    merged_df = merged_df.repartition(npartitions=partitions)

    basic_grouped_df = merged_df.groupby(
        ['category', 'high_value_transaction'],
        sort=False  # Disabling sorting speeds up the operation
    ).agg({
        'amount': 'sum',
        'transaction_id': 'count',
        'balance': 'mean'
    }).reset_index()

    # Execute the lazy chain for aggregations
    print("\nCollecting results...")
    collection_start_time = time.time()

    # Use a distributed client for efficiency
    try:
        client = Client.current() or Client(processes=False, threads_per_worker=4)
        print(f"Using dask client with {len(client.scheduler_info()['workers'])} workers")
        basic_result = basic_grouped_df.compute()
    except:
        print("Using regular compute (no distributed client)")
        basic_result = basic_grouped_df.compute()

    # Compute unique counts separately with optimizations
    print("Computing unique counts...")

    unique_account_df = merged_df[['category', 'high_value_transaction', 'account_id']].drop_duplicates()
    unique_merchant_df = merged_df[['category', 'high_value_transaction', 'merchant_id']].drop_duplicates()

    unique_accounts = unique_account_df.groupby(['category', 'high_value_transaction']).size().compute()
    unique_merchants = unique_merchant_df.groupby(['category', 'high_value_transaction']).size().compute()

    # Convert to DataFrames
    unique_accounts = unique_accounts.to_frame('unique_accounts')
    unique_merchants = unique_merchants.to_frame('unique_merchants')

    # Merge results
    result = basic_result.merge(unique_accounts, on=['category', 'high_value_transaction']) \
                         .merge(unique_merchants, on=['category', 'high_value_transaction'])

    collection_time = time.time() - collection_start_time
    print(f"Collection completed in {collection_time:.4f} seconds.")

    print("\nSample of the result:")
    print(result.head(5))

    # ------ Generate a summary report ------
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print(f"Memory after merging: {after_merging_memory:.2f} MB")
    print(f"Collection time: {collection_time:.4f} seconds.")

    # Save the final result
    result.to_parquet(f"{input_dir}/grouped_data.parquet")
    result.to_csv(f"{input_dir}/grouped_data.csv")

    return result

if __name__ == "__main__":
    perform_operations()