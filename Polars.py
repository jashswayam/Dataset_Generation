import polars as pl
import psutil
import time
import threading
import queue

# Global queue for thread communication
result_queue = queue.Queue()

def get_memory_usage():
    """
    Returns the current memory usage in MB.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Memory in MB

def track_performance(result_queue):
    """
    Track memory usage and time in a separate thread
    """
    # Memory before operations
    mem_before = get_memory_usage()
    result_queue.put(("mem_before", mem_before))
    
    # Track time before operations
    start_time = time.time()
    result_queue.put(("start_time", start_time))

def perform_operations(input_dir="bank_data_joins"):
    # Load the generated CSV data using Polars
    print("Loading data...")
    accounts_df = pl.read_csv(f"{input_dir}/accounts.csv")
    merchants_df = pl.read_csv(f"{input_dir}/merchants.csv")
    transactions_df = pl.read_csv(f"{input_dir}/transactions.csv")

    # Start the performance tracking in a separate thread
    performance_thread = threading.Thread(target=track_performance, args=(result_queue,))
    performance_thread.start()

    # ------ Filtering Operation ------
    print("Performing filtering operation...")
    filtered_accounts = accounts_df.filter(pl.col('balance') > 10000)  # filter accounts with balance > 10K

    # ------ Merging Operation ------
    print("Performing merging operation...")
    # In Polars, join is used instead of merge
    merged_df = filtered_accounts.join(transactions_df, on='account_id', how='inner')
    merged_df = merged_df.join(merchants_df, on='merchant_id', how='inner')

    # ------ Conditional Operation ------
    print("Performing conditional operation...")
    # In Polars, we can use with_columns for creating new columns
    merged_df = merged_df.with_columns(
        pl.when(pl.col('amount') > 500).then('Yes').otherwise('No').alias('high_value_transaction')
    )

    # Wait for the performance tracking thread to finish
    performance_thread.join()

    # Retrieve performance results from the queue
    mem_before = result_queue.get()[1]  # Extract the value from the tuple
    start_time = result_queue.get()[1]  # Extract the value from the tuple

    # Track memory and time after operations
    mem_after_filter = get_memory_usage()
    filter_time = time.time() - start_time
    print(f"Filtering operation completed in {filter_time:.4f} seconds.")
    print(f"Memory after filtering operation: {mem_after_filter:.2f} MB")

    mem_after_merge = get_memory_usage()
    merge_time = time.time() - start_time
    print(f"Merging operation completed in {merge_time:.4f} seconds.")
    print(f"Memory after merging operation: {mem_after_merge:.2f} MB")

    mem_after_condition = get_memory_usage()
    condition_time = time.time() - start_time
    print(f"Conditional operation completed in {condition_time:.4f} seconds.")
    print(f"Memory after conditional operation: {mem_after_condition:.2f} MB")

    # ------ Generate a report ------
    print("\n--- Report ---")
    print(f"Memory before operation: {mem_before:.2f} MB")
    print(f"Memory after filtering operation: {mem_after_filter:.2f} MB")
    print(f"Memory after merging operation: {mem_after_merge:.2f} MB")
    print(f"Memory after conditional operation: {mem_after_condition:.2f} MB")
    
    print(f"\nFiltering operation took: {filter_time:.4f} seconds.")
    print(f"Merging operation took: {merge_time:.4f} seconds.")
    print(f"Conditional operation took: {condition_time:.4f} seconds.")
    
    # Optionally save the final result to a new CSV file
    merged_df.write_csv(f"{input_dir}/merged_data.csv")
    
    return merged_df

if __name__ == "__main__":
    perform_operations()
