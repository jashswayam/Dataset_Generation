from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count as spark_count, avg, lit, when, countDistinct
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
    
    # Create a Spark session
    print("\nCreating Spark session and loading data...")
    spark = SparkSession.builder \
        .appName("BankDataAnalysis") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    # Load the generated data
    accounts_df = spark.read.parquet(f"{input_dir}/accounts.parquet")
    merchants_df = spark.read.parquet(f"{input_dir}/merchants.parquet")
    transactions_df = spark.read.parquet(f"{input_dir}/transactions.parquet")
    
    # Cache the DataFrames to improve performance
    accounts_df.cache()
    merchants_df.cache()
    transactions_df.cache()
    
    # Memory after loading
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for loading: {loading_memory_diff:.2f} MB")
    print(f"Total memory after loading: {loading_memory:.2f} MB")
    
    # ------ Filtering Operation (Standalone) ------
    print("\nPerforming filtering operation (Standalone)...")
    start_filter_time = time.time()
    before_filter_memory = get_memory_usage()
    
    # Standalone filtering for accounts
    print("Filtering accounts...")
    filtered_accounts = accounts_df.filter(
        (col('balance') > 10000) & 
        (col('status') == 'Active')
    )
    
    # Count rows to materialize the result
    filtered_count = filtered_accounts.count()
    print(f"Filtered accounts count: {filtered_count}")
    
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
    merged_df = merged_df.withColumn(
        'high_value_transaction',
        when(col('amount') > 500, 'Yes').otherwise('No')
    )
    
    # Cache the merged DataFrame
    merged_df.cache()
    
    # Materialize the merged DataFrame
    merged_count = merged_df.count()
    print(f"Merged rows count: {merged_count}")
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after merging
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
    
    # Define a groupby operation
    grouped_df = merged_df.groupBy([
        'category',  # Using merchant category
        'high_value_transaction'
    ]).agg(
        spark_sum('amount').alias('total_amount'),
        spark_count('transaction_id').alias('transaction_count'),
        avg('balance').alias('avg_balance'),
        countDistinct('account_id').alias('unique_accounts'),
        countDistinct('merchant_id').alias('unique_merchants')
    )
    
    # Execute the groupby and materialize the results
    print("\nCollecting results...")
    collection_start_time = time.time()
    result = grouped_df.collect()
    collection_time = time.time() - collection_start_time
    print(f"Collection completed in {collection_time:.4f} seconds.")
    
    # Convert the result to a DataFrame and display sample
    result_df = spark.createDataFrame(result)
    print("\nSample of the result:")
    result_df.show(5)
    
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
    print(f"Memory used for loading: {loading_memory_diff:.2f} MB")
    print(f"Memory used by filtering operation (standalone): {filter_memory_diff:.2f} MB")
    print(f"Memory used by merging operation (separate): {merging_memory_diff:.2f} MB")
    print(f"Memory used by group by operation: {groupby_memory_diff:.2f} MB")
    print(f"Total memory increase: {after_groupby_memory - initial_memory:.2f} MB")
    
    print(f"\nFiltering operation took: {filter_time:.4f} seconds.")
    print(f"Merging operation took: {merge_time:.4f} seconds.")
    print(f"Group by operation took: {groupby_time:.4f} seconds.")
    print(f"Collection time: {collection_time:.4f} seconds.")
    print(f"Total processing time: {time.time() - start_filter_time:.4f} seconds.")
    
    # Save the final result to parquet and CSV
    result_df.write.mode("overwrite").parquet(f"{input_dir}/grouped_data.parquet")
    result_df.write.mode("overwrite").option("header", "true").csv(f"{input_dir}/grouped_data_csv")
    
    # Stop the Spark session
    spark.stop()
    
    return result_df

if __name__ == "__main__":
    perform_operations()