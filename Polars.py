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
    
    # Load the generated CSV data with lazy execution
    print("\nLazy loading data...")
    accounts_df = pl.scan_csv(f"{input_dir}/accounts.csv")
    merchants_df = pl.scan_csv(f"{input_dir}/merchants.csv")
    transactions_df = pl.scan_csv(f"{input_dir}/transactions.csv")

    # Memory after setting up lazy loading
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for lazy loading setup: {loading_memory_diff:.2f} MB")
    print(f"Total memory after lazy loading setup: {loading_memory:.2f} MB")
    
    # Start tracking time
    start_time = time.time()

    # ------ Merging Operation with Conditional Filtering ------
    print("\nPerforming merging operation with conditional filtering...")
    before_merging_memory = get_memory_usage()
    
    # Merge with conditional filtering within the lazy execution chain
    merged_df = (accounts_df
                .filter(pl.col('balance') > 10000)
                .join(transactions_df, on='account_id', how='inner')
                .join(merchants_df, on='merchant_id', how='inner')
                .with_columns(
                    pl.when(pl.col('amount') > 500).then(pl.lit('Yes')).otherwise(pl.lit('No')).alias('high_value_transaction')
                ))
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after merging setup (still lazy)
    after_merging_memory = get_memory_usage()
    merging_memory_diff = after_merging_memory - before_merging_memory
    merge_time = time.time() - start_time
    print(f"Merging operation setup completed in {merge_time:.4f} seconds.")
    print(f"Memory used by merging operation setup: {merging_memory_diff:.2f} MB")
    print(f"Total memory after merging setup: {after_merging_memory:.2f} MB")

    # ------ Group By Operation ------
    print("\nPerforming group by operation...")
    before_groupby_memory = get_memory_usage()
    
    # Define a groupby operation in the lazy chain
    grouped_df = (merged_df
                 .groupby(['merchant_category', 'high_value_transaction'])
                 .agg([
                     pl.sum('amount').alias('total_amount'),
                     pl.count('transaction_id').alias('transaction_count'),
                     pl.mean('balance').alias('avg_balance')
                 ]))
    
    # Execute the lazy chain and materialize the results
    result = grouped_df.collect()
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after groupby execution
    after_groupby_memory = get_memory_usage()
    groupby_memory_diff = after_groupby_memory - before_groupby_memory
    groupby_time = time.time() - start_time
    print(f"Group by operation completed in {groupby_time:.4f} seconds.")
    print(f"Memory used by group by operation: {groupby_memory_diff:.2f} MB")
    print(f"Total memory after group by: {after_groupby_memory:.2f} MB")

    # ------ Generate a summary report ------
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print(f"Memory used for lazy loading setup: {loading_memory_diff:.2f} MB")
    print(f"Memory used by merging operation with filtering: {merging_memory_diff:.2f} MB")
    print(f"Memory used by group by operation: {groupby_memory_diff:.2f} MB")
    print(f"Total memory increase: {after_groupby_memory - initial_memory:.2f} MB")
    
    print(f"\nMerging operation setup took: {merge_time:.4f} seconds.")
    print(f"Group by operation execution took: {groupby_time - merge_time:.4f} seconds.")
    print(f"Total processing time: {groupby_time:.4f} seconds.")
    
    # Optionally save the final result to a new CSV file
    result.write_csv(f"{input_dir}/grouped_data.csv")

if __name__ == "__main__":
    perform_operations()
