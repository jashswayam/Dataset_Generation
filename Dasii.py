import dask.dataframe as dd
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
    
    # Load the generated data with lazy execution
    print("\nLazy loading data...")
    # Using parquet files instead of CSV for better performance
    accounts_df = dd.read_parquet(f"{input_dir}/accounts.parquet")
    merchants_df = dd.read_parquet(f"{input_dir}/merchants.parquet")
    transactions_df = dd.read_parquet(f"{input_dir}/transactions.parquet")

    # Memory after setting up lazy loading
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for lazy loading setup: {loading_memory_diff:.2f} MB")
    print(f"Total memory after lazy loading setup: {loading_memory:.2f} MB")
    
    # ------ Filtering Operation (Standalone) ------
    print("\nPerforming filtering operation (Standalone)...")
    start_filter_time = time.time()
    before_filter_memory = get_memory_usage()
    
    # Standalone filtering for accounts
    print("Filtering accounts...")
    filtered_accounts = accounts_df[(accounts_df['balance'] > 10000) & 
                                    (accounts_df['status'] == 'Active')]
    
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
    accounts_transactions = accounts_df.merge(transactions_df, 
                                              left_on='account_id', 
                                              right_on='account_id', 
                                              how='inner')
    
    # Then join with merchants
    print("Joining with merchants...")
    merged_df = accounts_transactions.merge(merchants_df, 
                                            left_on='merchant_id', 
                                            right_on='merchant_id', 
                                            how='inner')
    
    # Add derived column
    merged_df['high_value_transaction'] = merged_df['amount'].apply(lambda x: 'Yes' if x > 500 else 'No', meta='str')
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after merging setup (still lazy)
    after_merging_memory = get_memory_usage()
    merging_memory_diff = after_merging_memory - before_merging_memory
    merge_time = time.time() - start_merge_time
    print(f"Merging operation setup completed in {merge_time:.4f} seconds.")
    print(f"Memory used by merging operation setup: {merging_memory_diff:.2f} MB")
    print(f"Total memory after merging setup: {after_merging_memory:.2f} MB")

    # ------ Group By Operation ------
    print("\nPerforming group by operation...")
    start_groupby_time = time.time()
    before_groupby_memory = get_memory_usage()
    
    # Define a groupby operation in the lazy chain with optimizations
    print("Performing optimized standard aggregations...")
    
    # 1. Repartition to optimize for groupby
    print("Repartitioning data for better groupby performance...")
    partitions = min(merged_df.npartitions, 32)  # Limit number of partitions
    merged_df = merged_df.repartition(npartitions=partitions)
    
    # 2. Set reasonably sized partitions for the groupby
    basic_grouped_df = merged_df.groupby(
        ['category', 'high_value_transaction'],
        sort=False,  # Turn off sorting for better performance
    ).agg({
        'amount': 'sum',
        'transaction_id': 'count',
        'balance': 'mean'
    }).reset_index()
    
    # Execute the lazy chain for basic aggregations
    print("\nCollecting basic aggregation results...")
    collection_start_time = time.time()
    
    # 3. Optimize computation with more workers if available
    from distributed import Client
    try:
        # Try to use existing client or create a temporary one
        client = Client.current() or Client(processes=False, threads_per_worker=4)
        print(f"Using dask client with {len(client.scheduler_info()['workers'])} workers")
        basic_result = basic_grouped_df.compute()
    except:
        # Fall back to regular compute if distributed setup fails
        print("Using regular compute (no distributed client)")
        basic_result = basic_grouped_df.compute()
    
    # Now handle unique counts separately with optimizations
    print("Computing unique counts with optimizations...")
    
    # Pre-compute drop_duplicates for better performance
    unique_account_df = merged_df[['category', 'high_value_transaction', 'account_id']].drop_duplicates()
    unique_merchant_df = merged_df[['category', 'high_value_transaction', 'merchant_id']].drop_duplicates()
    
    # Count unique values after deduplication
    unique_accounts = unique_account_df.groupby(['category', 'high_value_transaction']).size().compute()
    unique_merchants = unique_merchant_df.groupby(['category', 'high_value_transaction']).size().compute()
    
    # Convert to DataFrames with proper column names
    unique_accounts = unique_accounts.to_frame('unique_accounts')
    unique_merchants = unique_merchants.to_frame('unique_merchants')
    
    # Merge the results
    result = basic_result.merge(
        unique_accounts, 
        on=['category', 'high_value_transaction']
    ).merge(
        unique_merchants,
        on=['category', 'high_value_transaction']
    )
    
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
    print(f"Memory used by filtering operation (standalone): {filter_memory_diff:.2f} MB")
    print(f"Memory used by merging operation (separate): {merging_memory_diff:.2f} MB")
    print(f"Memory used by group by operation: {groupby_memory_diff:.2f} MB")
    print(f"Total memory increase: {after_groupby_memory - initial_memory:.2f} MB")
    
    print(f"\nFiltering operation took: {filter_time:.4f} seconds.")
    print(f"Merging operation took: {merge_time:.4f} seconds.")
    print(f"Group by operation took: {groupby_time:.4f} seconds.")
    print(f"Collection time: {collection_time:.4f} seconds.")
    print(f"Total processing time: {time.time() - start_filter_time:.4f} seconds.")
    
    # Optionally save the final result to a new parquet file
    result.to_parquet(f"{input_dir}/grouped_data.parquet")
    result.to_csv(f"{input_dir}/grouped_data.csv")
    
    return result

if __name__ == "__main__":
    perform_operations()l