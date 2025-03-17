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
    
    # Define a groupby operation in the lazy chain
    # Use 'nunique' properly with Dask
    grouped_df = merged_df.groupby(['category', 'high_value_transaction']).agg({
        'amount': 'sum',
        'transaction_id': 'count',
        'balance': 'mean',
        'account_id': 'nunique',  # Dask supports nunique directly
        'merchant_id': 'nunique'
    }).reset_index()  # Reset index to make it a flat DataFrame for easier display
    
    # Execute the lazy chain and materialize the results
    print("\nCollecting results...")
    collection_start_time = time.time()
    result = grouped_df.compute()  # This triggers the actual computation
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
    perform_operations()
