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
    
    # Load the generated CSV data
    print("\nLoading data...")
    accounts_df = pl.read_csv(f"{input_dir}/accounts.csv")
    merchants_df = pl.read_csv(f"{input_dir}/merchants.csv")
    transactions_df = pl.read_csv(f"{input_dir}/transactions.csv")

    # Memory after loading data
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for loading data: {loading_memory_diff:.2f} MB")
    print(f"Total memory after loading: {loading_memory:.2f} MB")
    
    # Start tracking time
    start_time = time.time()

    # ------ Filtering Operation ------
    print("\nPerforming filtering operation...")
    before_filtering_memory = get_memory_usage()
    filtered_accounts = accounts_df.filter(pl.col('balance') > 10000)
    
    # Force garbage collection to get accurate memory difference
    gc.collect()
    
    # Track memory and time after filtering
    after_filtering_memory = get_memory_usage()
    filtering_memory_diff = after_filtering_memory - before_filtering_memory
    filter_time = time.time() - start_time
    print(f"Filtering operation completed in {filter_time:.4f} seconds.")
    print(f"Memory used by filtering operation: {filtering_memory_diff:.2f} MB")
    print(f"Total memory after filtering: {after_filtering_memory:.2f} MB")

    # ------ Merging Operation ------
    print("\nPerforming merging operation...")
    before_merging_memory = get_memory_usage()
    merged_df = filtered_accounts.join(transactions_df, on='account_id', how='inner')
    merged_df = merged_df.join(merchants_df, on='merchant_id', how='inner')
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after merging
    after_merging_memory = get_memory_usage()
    merging_memory_diff = after_merging_memory - before_merging_memory
    merge_time = time.time() - start_time
    print(f"Merging operation completed in {merge_time:.4f} seconds.")
    print(f"Memory used by merging operation: {merging_memory_diff:.2f} MB")
    print(f"Total memory after merging: {after_merging_memory:.2f} MB")

    # ------ Conditional Operation ------
    print("\nPerforming conditional operation...")
    before_conditional_memory = get_memory_usage()
    merged_df = merged_df.with_columns(
        pl.when(pl.col('amount') > 500).then(pl.lit('Yes')).otherwise(pl.lit('No')).alias('high_value_transaction')
    )
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after conditional operations
    after_conditional_memory = get_memory_usage()
    conditional_memory_diff = after_conditional_memory - before_conditional_memory
    condition_time = time.time() - start_time
    print(f"Conditional operation completed in {condition_time:.4f} seconds.")
    print(f"Memory used by conditional operation: {conditional_memory_diff:.2f} MB")
    print(f"Total memory after conditional operation: {after_conditional_memory:.2f} MB")

    # ------ Generate a summary report ------
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print(f"Memory used for loading data: {loading_memory_diff:.2f} MB")
    print(f"Memory used by filtering operation: {filtering_memory_diff:.2f} MB")
    print(f"Memory used by merging operation: {merging_memory_diff:.2f} MB")
    print(f"Memory used by conditional operation: {conditional_memory_diff:.2f} MB")
    print(f"Total memory increase: {after_conditional_memory - initial_memory:.2f} MB")
    
    print(f"\nFiltering operation took: {filter_time:.4f} seconds.")
    print(f"Merging operation took: {merge_time - filter_time:.4f} seconds.")
    print(f"Conditional operation took: {condition_time - merge_time:.4f} seconds.")
    print(f"Total processing time: {condition_time:.4f} seconds.")
    
    # Optionally save the final result to a new CSV file
    merged_df.write_csv(f"{input_dir}/merged_data.csv")

if __name__ == "__main__":
    perform_operations()
