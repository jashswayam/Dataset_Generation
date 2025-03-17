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
    """
    Returns the current memory usage in MB.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Memory in MB

# Define Numba-accelerated functions for aggregation
@jit(nopython=True, parallel=True)
def numba_sum(arr):
    return np.sum(arr)

@jit(nopython=True, parallel=True)
def numba_mean(arr):
    return np.mean(arr)

@jit(nopython=True, parallel=True)
def numba_count_unique(arr):
    # Create a set of unique values using Numba's approach
    unique_set = set()
    for x in arr:
        unique_set.add(x)
    return len(unique_set)

@jit(nopython=True)
def numba_filter_high_value(amounts, threshold=500):
    """Return array of Yes/No strings based on amount threshold"""
    result = np.empty(len(amounts), dtype=numba.types.string)
    for i in range(len(amounts)):
        if amounts[i] > threshold:
            result[i] = 'Yes'
        else:
            result[i] = 'No'
    return result

# Custom aggregation function using Numba
def fast_group_aggregate(df, group_cols, agg_cols):
    """
    Perform fast group by aggregation using Numba-accelerated functions
    """
    # Convert to numpy arrays for Numba
    df_dict = {col: df[col].values for col in df.columns}
    
    # Get unique group combinations
    group_values = df[group_cols].drop_duplicates()
    
    # Prepare result containers
    results = []
    
    # For each unique group combination
    for _, group_row in group_values.iterrows():
        # Create mask for this group
        mask = np.ones(len(df), dtype=bool)
        for col in group_cols:
            mask = mask & (df_dict[col] == group_row[col])
        
        # Skip if no data for this group
        if not np.any(mask):
            continue
        
        # Create result row starting with group values
        result_row = {col: group_row[col] for col in group_cols}
        
        # Calculate aggregations using Numba
        for col in agg_cols:
            # Only get values for this group
            group_data = df_dict[col][mask]
            
            # Skip if no data
            if len(group_data) == 0:
                continue
                
            # Calculate sums
            if col == 'amount':
                result_row[f'{col}_sum'] = numba_sum(group_data)
                result_row[f'{col}_mean'] = numba_mean(group_data)
                result_row[f'{col}_count'] = len(group_data)
            elif col == 'balance':
                result_row[f'{col}_mean'] = numba_mean(group_data)
            elif col == 'account_id' or col == 'merchant_id':
                # Count unique values
                result_row[f'{col}_unique'] = len(np.unique(group_data))
                
        results.append(result_row)
    
    # Convert results to DataFrame
    return pd.DataFrame(results)

def perform_operations(input_dir="bank_data_joins"):
    # Initial memory measurement
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Load the generated data with lazy execution and optimized reading
    print("\nLazy loading data with optimizations...")
    
    # Use pyarrow engine for better performance and read only needed columns
    accounts_df = dd.read_parquet(
        f"{input_dir}/accounts.parquet",
        engine='pyarrow',
        columns=['account_id', 'balance', 'status']
    )
    
    merchants_df = dd.read_parquet(
        f"{input_dir}/merchants.parquet",
        engine='pyarrow',
        columns=['merchant_id', 'category']
    )
    
    transactions_df = dd.read_parquet(
        f"{input_dir}/transactions.parquet",
        engine='pyarrow',
        columns=['transaction_id', 'account_id', 'merchant_id', 'amount']
    )

    # Memory after setting up lazy loading
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for lazy loading setup: {loading_memory_diff:.2f} MB")
    print(f"Total memory after lazy loading setup: {loading_memory:.2f} MB")

    # ------ Start Pipeline with Numba optimizations ------
    print("\nPerforming Numba-optimized pipeline...")
    start_pipeline_time = time.time()
    before_pipeline_memory = get_memory_usage()

    # Pre-filter in Dask to reduce data size
    accounts_filtered = accounts_df[accounts_df['status'] == 'Active']
    
    # Process in chunks using Dask's partitioning
    chunk_results = []
    
    # Get accounts and merchants in pandas form before processing
    # This avoids loading them repeatedly for each chunk
    print("Converting accounts and merchants to pandas...")
    accounts_pd = accounts_filtered.compute()
    merchants_pd = merchants_df.compute()
    
    @delayed
    def process_chunk(txn_part):
        # Convert partition to pandas
        txn_pd = txn_part.compute()
        
        # First join transactions with merchants
        merged = pd.merge(
            txn_pd, 
            merchants_pd,
            on='merchant_id',
            how='inner'
        )
        
        # Then join with accounts
        merged = pd.merge(
            merged,
            accounts_pd,
            on='account_id',
            how='inner'
        )
        
        # Add high value flag using Numba
        amounts = merged['amount'].values
        merged['high_value_transaction'] = numba_filter_high_value(amounts)
        
        # Perform fast group-by using Numba
        result = fast_group_aggregate(
            merged,
            group_cols=['category', 'high_value_transaction'],
            agg_cols=['amount', 'balance', 'account_id', 'merchant_id']
        )
        
        return result
    
    # Process each partition
    print("Processing data in chunks with Numba acceleration...")
    for i in range(min(transactions_df.npartitions, 10)):  # Limit to 10 partitions for memory management
        # Get partition
        chunk_result = process_chunk(
            transactions_df.get_partition(i)
        )
        chunk_results.append(chunk_result)
    
    # Combine results
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
    
    # Execute with progress bar
    print("\nComputing final results...")
    collection_start_time = time.time()
    with ProgressBar():
        result = final_result_delayed.compute()
    
    collection_time = time.time() - collection_start_time
    print(f"Collection completed in {collection_time:.4f} seconds.")
    
    # Display sample of the result
    print("\nSample of the result:")
    print(result.head(5))
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after execution
    after_pipeline_memory = get_memory_usage()
    pipeline_memory_diff = after_pipeline_memory - before_pipeline_memory
    pipeline_time = time.time() - start_pipeline_time
    print(f"Pipeline completed in {pipeline_time:.4f} seconds.")
    print(f"Memory used by pipeline: {pipeline_memory_diff:.2f} MB")
    print(f"Total memory after pipeline: {after_pipeline_memory:.2f} MB")
    
    # ------ Generate a summary report ------
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print(f"Memory used for lazy loading setup: {loading_memory_diff:.2f} MB")
    print(f"Memory used by Numba pipeline: {pipeline_memory_diff:.2f} MB")
    print(f"Total memory increase: {after_pipeline_memory - initial_memory:.2f} MB")
    
    print(f"\nNumba pipeline took: {pipeline_time:.4f} seconds.")
    print(f"Collection time: {collection_time:.4f} seconds.")
    print(f"Total processing time: {time.time() - start_pipeline_time:.4f} seconds.")
    
    # Optionally save the final result to a new parquet file with compression
    result.to_parquet(f"{input_dir}/grouped_data_numba.parquet", compression='snappy')
    result.to_csv(f"{input_dir}/grouped_data_numba.csv")
    
    return result

if __name__ == "__main__":
    perform_operations()