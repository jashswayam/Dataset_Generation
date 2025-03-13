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
    
    # Load the data with Dask
    print("\nLoading data with Dask...")
    
    accounts_df = dd.read_parquet(f"{input_dir}/accounts.parquet")
    merchants_df = dd.read_parquet(f"{input_dir}/merchants.parquet")
    transactions_df = dd.read_parquet(f"{input_dir}/transactions.parquet")
    
    # Memory after loading
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for loading with Dask: {loading_memory_diff:.2f} MB")
    print(f"Total memory after loading: {loading_memory:.2f} MB")
    
    # ------ Filtering Operation ------
    print("\nPerforming filtering operation...")
    start_filter_time = time.time()
    
    filtered_accounts = accounts_df[(accounts_df['balance'] > 10000) & 
                                    (accounts_df['status'] == 'Active')]
    
    # Trigger computation
    filtered_accounts = filtered_accounts.compute()
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after filtering
    after_filter_memory = get_memory_usage()
    filter_memory_diff = after_filter_memory - loading_memory
    filter_time = time.time() - start_filter_time
    print(f"Filtering completed in {filter_time:.4f} seconds.")
    print(f"Memory used by filtering: {filter_memory_diff:.2f} MB")
    print(f"Total memory after filtering: {after_filter_memory:.2f} MB")
    
    # ------ Merging Operation ------
    print("\nPerforming merging operation...")
    start_merge_time = time.time()
    
    # Join accounts with transactions
    accounts_transactions = accounts_df.merge(transactions_df, on='account_id', how='inner')
    
    # Join with merchants
    merged_df = accounts_transactions.merge(merchants_df, on='merchant_id', how='inner')
    
    # Add derived column
    merged_df['high_value_transaction'] = merged_df['amount'].map(lambda x: 'Yes' if x > 500 else 'No', meta=('amount', 'object'))
    
    # Compute the final merged dataframe
    merged_df = merged_df.compute()
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after merging
    after_merging_memory = get_memory_usage()
    merging_memory_diff = after_merging_memory - after_filter_memory
    merge_time = time.time() - start_merge_time
    print(f"Merging completed in {merge_time:.4f} seconds.")
    print(f"Memory used by merging: {merging_memory_diff:.2f} MB")
    print(f"Total memory after merging: {after_merging_memory:.2f} MB")

    # ------ Group By Operation ------
    print("\nPerforming group by operation...")
    start_groupby_time = time.time()
    
    # Convert back to Dask DataFrame for groupby
    merged_ddf = dd.from_pandas(merged_df, npartitions=4)
    
    grouped_ddf = merged_ddf.groupby(['category', 'high_value_transaction']).agg({
    'amount': 'sum',
    'transaction_id': 'count',
    'balance': 'mean'
})

grouped_ddf = grouped_ddf.compute()
grouped_ddf['account_id_nunique'] = merged_df['account_id'].nunique()
grouped_ddf['merchant_id_nunique'] = merged_df['merchant_id'].nunique()
grouped_df = grouped_ddf.reset_index()
    
    # Compute final grouped result
    grouped_df = grouped_df.compute()
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after groupby execution
    after_groupby_memory = get_memory_usage()
    groupby_memory_diff = after_groupby_memory - after_merging_memory
    groupby_time = time.time() - start_groupby_time
    print(f"Group by completed in {groupby_time:.4f} seconds.")
    print(f"Memory used by group by: {groupby_memory_diff:.2f} MB")
    print(f"Total memory after group by: {after_groupby_memory:.2f} MB")

    # ------ Generate a summary report ------
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print(f"Memory used for loading with Dask: {loading_memory_diff:.2f} MB")
    print(f"Memory used by filtering: {filter_memory_diff:.2f} MB")
    print(f"Memory used by merging: {merging_memory_diff:.2f} MB")
    print(f"Memory used by group by: {groupby_memory_diff:.2f} MB")
    print(f"Total memory increase: {after_groupby_memory - initial_memory:.2f} MB")
    
    print(f"\nFiltering took: {filter_time:.4f} seconds.")
    print(f"Merging took: {merge_time:.4f} seconds.")
    print(f"Group by took: {groupby_time:.4f} seconds.")
    print(f"Total processing time: {time.time() - start_filter_time:.4f} seconds.")
    
    # Save the final results
    grouped_df.to_parquet(f"{input_dir}/grouped_data_dask.parquet")
    grouped_df.to_csv(f"{input_dir}/grouped_data_dask.csv", index=False)
    
    return grouped_df

if __name__ == "__main__":
    perform_operations()
