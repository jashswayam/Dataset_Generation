import polars as pl
import psutil
import time
import gc

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
    
    # Load the generated data with eager execution
    print("\nEager loading data...")
    # Using parquet files with eager loading
    accounts_df = pl.read_parquet(f"{input_dir}/accounts.parquet")
    merchants_df = pl.read_parquet(f"{input_dir}/merchants.parquet")
    transactions_df = pl.read_parquet(f"{input_dir}/transactions.parquet")

    # Memory after eager loading
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for eager loading: {loading_memory_diff:.2f} MB")
    print(f"Total memory after eager loading: {loading_memory:.2f} MB")
    
    # ------ Filtering Operation (Standalone) ------
    print("\nPerforming filtering operation (Standalone)...")
    start_filter_time = time.time()
    before_filter_memory = get_memory_usage()
    
    # Standalone filtering for accounts
    print("Filtering accounts...")
    filtered_accounts = accounts_df.filter(
        (pl.col('balance') > 10000) & 
        (pl.col('status') == 'Active')
    )
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after filtering
    after_filter_memory = get_memory_usage()
    filter_memory_diff = after_filter_memory - before_filter_memory
    filter_time = time.time() - start_filter_time
    print(f"Filtering operations completed in {filter_time:.4f} seconds.")
    print(f"Memory used by filtering operations: {filter_memory_diff:.2f} MB")
    print(f"Total memory after filtering: {after_filter_memory:.2f} MB")
    
    # ------ Merging Operation (Completely Separate) ------
    print("\nPerforming merging operation (without filtering)...")
    start_merge_time = time.time()
    before_merging_memory = get_memory_usage()
    
    # Start fresh with original dataframes for merging
    # First join accounts with transactions
    print("Joining accounts with transactions...")
    accounts_transactions = accounts_df.join(
        transactions_df, 
        on='account_id', 
        how='inner'
    )
    
    # Then join with merchants
    print("Joining with merchants...")
    merged_df = accounts_transactions.join(
        merchants_df, 
        on='merchant_id', 
        how='inner'
    )
    
    # Add derived column
    merged_df = merged_df.with_columns(
        pl.when(pl.col('amount') > 500)
          .then(pl.lit('Yes'))
          .otherwise(pl.lit('No'))
          .alias('high_value_transaction')
    )
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after merging (immediately materialized in eager mode)
    after_merging_memory = get_memory_usage()
    merging_memory_diff = after_merging_memory - before_merging_memory
    merge_time = time.time() - start_merge_time
    print(f"Merging operation completed in {merge_time:.4f} seconds.")
    print(f"Memory used by merging operation: {merging_memory_diff:.2f} MB")
    print(f"Total memory after merging: {after_merging_memory:.2f} MB")

    # ------ Group By Operation ------
    print("\nPerforming group by operation...")
    start_groupby_time = time.time()
    before_groupby_memory = get_memory_usage()
    
    # Define and execute a groupby operation
    grouped_df = merged_df.group_by([
        'category',  # Using merchant category
        'high_value_transaction'
    ]).agg([
        pl.sum('amount').alias('total_amount'),
        pl.count('transaction_id').alias('transaction_count'),
        pl.mean('balance').alias('avg_balance'),
        pl.n_unique('account_id').alias('unique_accounts'),
        pl.n_unique('merchant_id').alias('unique_merchants')
    ])
    
    # No separate collection step needed in eager mode
    
    # Display sample of the result
    print("\nSample of the result:")
    print(grouped_df.head(5))
    
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
    print(f"Memory used for eager loading: {loading_memory_diff:.2f} MB")
    print(f"Memory used by filtering operation (standalone): {filter_memory_diff:.2f} MB")
    print(f"Memory used by merging operation: {merging_memory_diff:.2f} MB")
    print(f"Memory used by group by operation: {groupby_memory_diff:.2f} MB")
    print(f"Total memory increase: {after_groupby_memory - initial_memory:.2f} MB")
    
    print(f"\nFiltering operation took: {filter_time:.4f} seconds.")
    print(f"Merging operation took: {merge_time:.4f} seconds.")
    print(f"Group by operation took: {groupby_time:.4f} seconds.")
    print(f"Total processing time: {time.time() - start_filter_time:.4f} seconds.")
    
    # Optionally save the final result to a new parquet file
    grouped_df.write_parquet(f"{input_dir}/grouped_data.parquet")
    grouped_df.write_csv(f"{input_dir}/grouped_data.csv")
    
    return grouped_df

if __name__ == "__main__":
    perform_operations()
