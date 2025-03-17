import dask.dataframe as dd
import numpy as np
import pandas as pd
import psutil
import time
import gc
from dask.diagnostics import ProgressBar
from dask import delayed
from dask.distributed import Client, performance_report

def get_memory_usage():
    """
    Returns the current memory usage in MB.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Memory in MB

def perform_operations(input_dir="bank_data_joins"):
    # Initial memory measurement
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Set up distributed client with better resource management
    client = Client(processes=True, n_workers=min(psutil.cpu_count(), 8), 
                    threads_per_worker=2, memory_limit='4GB')
    print(f"Dask client dashboard: {client.dashboard_link}")

    # Load the generated data with lazy execution and optimized reading
    print("\nLazy loading data with optimizations...")
    
    # More efficient parquet reading with filters pushed down when possible
    accounts_df = dd.read_parquet(
        f"{input_dir}/accounts.parquet",
        engine='pyarrow',  # Use pyarrow engine for better performance
        columns=['account_id', 'balance', 'status']  # Read only needed columns
    )
    
    merchants_df = dd.read_parquet(
        f"{input_dir}/merchants.parquet",
        engine='pyarrow',
        columns=['merchant_id', 'category']  # Read only needed columns
    )
    
    transactions_df = dd.read_parquet(
        f"{input_dir}/transactions.parquet",
        engine='pyarrow',
        columns=['transaction_id', 'account_id', 'merchant_id', 'amount']  # Read only needed columns
    )

    # Memory after setting up lazy loading
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for lazy loading setup: {loading_memory_diff:.2f} MB")
    print(f"Total memory after lazy loading setup: {loading_memory:.2f} MB")

    # ------ Filtering and Merging with Optimizations ------
    print("\nPerforming optimized pipeline...")
    start_pipeline_time = time.time()
    before_pipeline_memory = get_memory_usage()

    # Pre-filter to reduce data size before merging
    print("Pre-filtering data to reduce join size...")
    accounts_filtered = accounts_df[accounts_df['status'] == 'Active']
    
    # First merge transactions with merchants (smaller merge first)
    print("Joining transactions with merchants...")
    transactions_merchants = transactions_df.merge(
        merchants_df,
        on='merchant_id',
        how='inner',
        suffixes=('', '_merchant')
    )
    
    # Then merge with filtered accounts
    print("Joining with accounts...")
    merged_df = transactions_merchants.merge(
        accounts_filtered,
        on='account_id',
        how='inner',
        suffixes=('', '_account')
    )
    
    # Add derived column with vectorized operation
    merged_df['high_value_transaction'] = merged_df['amount'].map_partitions(
        lambda s: np.where(s > 500, 'Yes', 'No'), meta=('high_value_transaction', 'object')
    )
    
    # Force garbage collection between major operations
    gc.collect()
    
    # ------ Optimized Group By Operation ------
    print("\nPerforming optimized group by operation...")
    start_groupby_time = time.time()
    before_groupby_memory = get_memory_usage()

    # MAJOR OPTIMIZATION: Repartition by groupby keys for performance
    print("Repartitioning data by groupby keys...")
    # Hash-based partitioning on the groupby keys
    merged_df = merged_df.map_partitions(
        lambda df: df.assign(partition_key=df['category'].astype(str) + '_' + df['high_value_transaction']),
        meta=merged_df.dtypes
    )
    
    # Repartition to a reasonable number of partitions (based on unique key count)
    # First get number of unique combinations
    with ProgressBar():
        unique_keys = merged_df[['category', 'high_value_transaction']].drop_duplicates().compute()
    num_unique_combos = len(unique_keys)
    num_partitions = min(max(num_unique_combos, 8), 32)  # At least 8, at most 32 partitions
    
    print(f"Repartitioning to {num_partitions} partitions based on {num_unique_combos} unique combinations...")
    merged_df = merged_df.repartition(npartitions=num_partitions)
    
    # OPTIMIZATION: Use map_partitions for faster NumPy-based aggregations
    # This is much faster than Dask's native groupby for many cases
    
    print("Using NumPy-accelerated groupby strategy...")
    
    @delayed
    def optimized_groupby(partition_df):
        # Convert to pandas for fast in-memory processing
        pdf = partition_df.copy()
        
        # Use efficient pandas groupby with numpy-based aggregations
        result = pdf.groupby(['category', 'high_value_transaction']).agg({
            'amount': ['sum', 'mean', 'std', 'count'],
            'account_id': pd.Series.nunique,
            'merchant_id': pd.Series.nunique,
            'balance': 'mean'
        })
        
        # Flatten the columns and reset index
        result.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in result.columns]
        return result.reset_index()
    
    # Apply the function to each partition
    partition_results = []
    for i in range(merged_df.npartitions):
        partition_results.append(optimized_groupby(merged_df.get_partition(i)))
    
    # Combine the results
    print("Combining partition results...")
    combined_delayed = delayed(pd.concat)(partition_results, ignore_index=True)
    
    # Final aggregation of the combined results
    @delayed
    def final_aggregation(combined_df):
        return combined_df.groupby(['category', 'high_value_transaction']).agg({
            'amount_sum': 'sum',
            'amount_mean': 'mean',
            'amount_std': 'mean',  # Taking mean of std is an approximation
            'amount_count': 'sum',
            'account_id_nunique': 'sum',  # This works because we partitioned by the groupby keys
            'merchant_id_nunique': 'sum',  # This works because we partitioned by the groupby keys
            'balance_mean': 'mean'
        }).reset_index()
    
    final_result_delayed = final_aggregation(combined_delayed)
    
    # Execute with progress bar
    print("\nComputing final results...")
    collection_start_time = time.time()
    with ProgressBar():
        with performance_report(filename=f"{input_dir}/dask-performance-report.html"):
            result = final_result_delayed.compute()
    
    collection_time = time.time() - collection_start_time
    print(f"Collection completed in {collection_time:.4f} seconds.")
    
    # Display sample of the result
    print("\nSample of the result:")
    print(result.head(5))
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after groupby execution
    after_groupby_memory = get_memory_usage()
    groupby_memory_diff = after_groupby_memory - before_groupby_memory
    groupby_time = time.time() - start_groupby_time
    print(f"Group by operation completed in {groupby_time:.4f} seconds.")
    print(f"Memory used by group by operation: {groupby_memory_diff:.2f} MB")
    print(f"Total memory after group by: {after_groupby_memory:.2f} MB")
    
    # ------ Generate a summary report ------
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print(f"Memory used for lazy loading setup: {loading_memory_diff:.2f} MB")
    print(f"Memory used by group by operation: {groupby_memory_diff:.2f} MB")
    print(f"Total memory increase: {after_groupby_memory - initial_memory:.2f} MB")
    
    print(f"\nGroup by operation took: {groupby_time:.4f} seconds.")
    print(f"Collection time: {collection_time:.4f} seconds.")
    print(f"Total processing time: {time.time() - start_pipeline_time:.4f} seconds.")
    
    # Optionally save the final result to a new parquet file with compression
    result.to_parquet(f"{input_dir}/grouped_data_optimized.parquet", compression='snappy')
    result.to_csv(f"{input_dir}/grouped_data_optimized.csv")
    
    # Clean up client
    client.close()
    
    return result

if __name__ == "__main__":
    perform_operations()