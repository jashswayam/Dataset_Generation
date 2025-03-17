import polars as pl
import psutil
import time
import gc

def get_memory_usage():
    """Returns the current memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Memory in MB

def perform_operations(input_dir="bank_data_joins"):
    # Initial memory measurement
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Load the generated data with lazy execution
    print("\nLazy loading data...")
    accounts_df = pl.scan_parquet(f"{input_dir}/accounts.parquet")
    merchants_df = pl.scan_parquet(f"{input_dir}/merchants.parquet")
    transactions_df = pl.scan_parquet(f"{input_dir}/transactions.parquet")

    # Memory after setting up lazy loading
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for lazy loading setup: {loading_memory_diff:.2f} MB")
    
    # ------ Filtering Operation (Hybrid approach) ------
    print("\nPerforming filtering operation (Hybrid approach)...")
    start_filter_time = time.time()
    before_filter_memory = get_memory_usage()

    # For filtering, use lazy execution but collect immediately
    # This can be more efficient for certain filters
    filtered_accounts = accounts_df.filter(
        (pl.col('balance') > 10000) & 
        (pl.col('status') == 'Active')
    ).collect()  # Collect immediately after filtering

    # Clear any temporary objects
    gc.collect()

    after_filter_memory = get_memory_usage()
    filter_memory_diff = after_filter_memory - before_filter_memory
    filter_time = time.time() - start_filter_time
    print(f"Filtering operations completed in {filter_time:.4f} seconds.")
    print(f"Memory used by filtering operations: {filter_memory_diff:.2f} MB")
    
    # ------ Merging Operation (Keep lazy for joins) ------
    print("\nPerforming merging operation (lazy)...")
    start_merge_time = time.time()
    before_merging_memory = get_memory_usage()

    # Create a lazy join pipeline
    accounts_transactions = accounts_df.join(
        transactions_df, 
        on='account_id', 
        how='inner'
    )

    merged_df = accounts_transactions.join(
        merchants_df, 
        on='merchant_id', 
        how='inner'
    )

    merged_df = merged_df.with_columns(
        pl.when(pl.col('amount') > 500)
          .then(pl.lit('Yes'))
          .otherwise(pl.lit('No'))
          .alias('high_value_transaction')
    )

    # For joins, keeping it lazy is usually more efficient
    # We'll collect only if needed for debugging
    # sample_result = merged_df.limit(5).collect()
    
    gc.collect()
    after_merging_memory = get_memory_usage()
    merging_memory_diff = after_merging_memory - before_merging_memory
    merge_time = time.time() - start_merge_time
    print(f"Merging operation setup completed in {merge_time:.4f} seconds.")
    print(f"Memory used by merging operation setup: {merging_memory_diff:.2f} MB")
    
    # ------ Group By Operation (Hybrid approach) ------
    print("\nPerforming group by operation...")
    start_groupby_time = time.time()
    before_groupby_memory = get_memory_usage()

    # For complex groupby operations, we can collect the data first
    # then perform the groupby on the collected DataFrame
    # This might be more memory efficient in some cases
    
    # Option 1: Keep it lazy and collect at the end
    grouped_df = merged_df.group_by([
        'category',
        'high_value_transaction'
    ]).agg([
        pl.sum('amount').alias('total_amount'),
        pl.count('transaction_id').alias('transaction_count'),
        pl.mean('balance').alias('avg_balance'),
        pl.n_unique('account_id').alias('unique_accounts'),
        pl.n_unique('merchant_id').alias('unique_merchants')
    ])
    
    # Option 2: Materialize before groupby (uncomment if this proves more efficient)
    # merged_df_collected = merged_df.collect()
    # grouped_df = merged_df_collected.group_by([
    #    'category',
    #    'high_value_transaction'
    # ]).agg([
    #    pl.sum('amount').alias('total_amount'),
    #    pl.count('transaction_id').alias('transaction_count'),
    #    pl.mean('balance').alias('avg_balance'),
    #    pl.n_unique('account_id').alias('unique_accounts'),
    #    pl.n_unique('merchant_id').alias('unique_merchants')
    # ])

    print("\nCollecting results...")
    collection_start_time = time.time()
    result = grouped_df.collect()
    collection_time = time.time() - collection_start_time
    print(f"Collection completed in {collection_time:.4f} seconds.")

    # Display sample of the result
    print("\nSample of the result:")
    print(result.head(5))

    gc.collect()
    after_groupby_memory = get_memory_usage()
    groupby_memory_diff = after_groupby_memory - before_groupby_memory
    groupby_time = time.time() - start_groupby_time
    print(f"Group by operation completed in {groupby_time:.4f} seconds.")
    print(f"Memory used by group by operation: {groupby_memory_diff:.2f} MB")
    
    # ------ Generate a summary report ------
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print(f"Memory used for lazy loading setup: {loading_memory_diff:.2f} MB")
    print(f"Memory used by filtering operation (hybrid): {filter_memory_diff:.2f} MB")
    print(f"Memory used by merging operation (lazy): {merging_memory_diff:.2f} MB")
    print(f"Memory used by group by operation: {groupby_memory_diff:.2f} MB")
    print(f"Total memory increase: {after_groupby_memory - initial_memory:.2f} MB")

    print(f"\nFiltering operation took: {filter_time:.4f} seconds.")
    print(f"Merging operation took: {merge_time:.4f} seconds.")
    print(f"Group by operation took: {groupby_time:.4f} seconds.")
    print(f"Collection time: {collection_time:.4f} seconds.")
    print(f"Total processing time: {time.time() - start_filter_time:.4f} seconds.")

    # Save the final result
    result.write_parquet(f"{input_dir}/grouped_data.parquet")
    result.write_csv(f"{input_dir}/grouped_data.csv")

    return result

if __name__ == "__main__":
    perform_operations()