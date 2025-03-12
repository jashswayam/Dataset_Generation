import pandas as pd
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
    # Load the generated CSV data
    print("Loading data...")
    accounts_df = pd.read_csv(f"{input_dir}/accounts.csv")
    merchants_df = pd.read_csv(f"{input_dir}/merchants.csv")
    transactions_df = pd.read_csv(f"{input_dir}/transactions.csv")

    # Start the performance tracking in a separate thread
    performance_thread = threading.Thread(target=track_performance, args=(result_queue,))
    performance_thread.start()

    # ------ Filtering Operation ------
    print("Performing filtering operation...")
    filtered_accounts = accounts_df[accounts_df['balance'] > 10000]  # Example: filter accounts with balance > 10K

    # ------ Merging Operation ------
    print("Performing merging operation...")
    merged_df = pd.merge(filtered_accounts, transactions_df, on='account_id', how='inner')
    merged_df = pd.merge(merged_df, merchants_df, on='merchant_id', how='inner')

    # ------ Conditional Operation ------
    print("Performing conditional operation...")
    merged_df['high_value_transaction'] = merged_df['amount'].apply(lambda x: 'Yes' if x > 500 else 'No')

    # Wait for the performance tracking thread to finish
    performance_thread.join()

    # Retrieve performance results from the queue
    mem_before = result_queue.get()
    start_time = result_queue.get()

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
    merged_df.to_csv(f"{input_dir}/merged_data.csv", index=False)

if __name__ == "__main__":
    perform_operations()
