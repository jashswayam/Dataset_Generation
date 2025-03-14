import psutil
import time
import gc
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit

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
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("BankDataOperations") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .getOrCreate()
    
    # Load the generated CSV data
    print("\nLoading data...")
    accounts_df = spark.read.csv(f"{input_dir}/accounts.csv", header=True, inferSchema=True)
    merchants_df = spark.read.csv(f"{input_dir}/merchants.csv", header=True, inferSchema=True)
    transactions_df = spark.read.csv(f"{input_dir}/transactions.csv", header=True, inferSchema=True)

    # Cache the DataFrames to improve performance for multiple operations
    accounts_df.cache()
    merchants_df.cache()
    transactions_df.cache()
    
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
    filtered_accounts = accounts_df.filter(col("balance") > 10000)
    # Force execution with count() to measure accurate time
    filtered_count = filtered_accounts.count()
    
    # Force garbage collection to get accurate memory difference
    gc.collect()
    
    # Track memory and time after filtering
    after_filtering_memory = get_memory_usage()
    filtering_memory_diff = after_filtering_memory - before_filtering_memory
    filter_time = time.time() - start_time
    print(f"Filtering operation completed in {filter_time:.4f} seconds.")
    print(f"Filtered {filtered_count} accounts.")
    print(f"Memory used by filtering operation: {filtering_memory_diff:.2f} MB")
    print(f"Total memory after filtering: {after_filtering_memory:.2f} MB")

    # ------ Merging Operation ------
    print("\nPerforming merging operation...")
    before_merging_memory = get_memory_usage()
    merged_df = filtered_accounts.join(transactions_df, "account_id", "inner")
    merged_df = merged_df.join(merchants_df, "merchant_id", "inner")
    # Force execution to measure accurate time
    merged_count = merged_df.count()
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after merging
    after_merging_memory = get_memory_usage()
    merging_memory_diff = after_merging_memory - before_merging_memory
    merge_time = time.time() - start_time
    print(f"Merging operation completed in {merge_time:.4f} seconds.")
    print(f"Merged dataframe has {merged_count} rows.")
    print(f"Memory used by merging operation: {merging_memory_diff:.2f} MB")
    print(f"Total memory after merging: {after_merging_memory:.2f} MB")

    # ------ Conditional Operation ------
    print("\nPerforming conditional operation...")
    before_conditional_memory = get_memory_usage()
    merged_df = merged_df.withColumn(
        "high_value_transaction", 
        when(col("amount") > 500, lit("Yes")).otherwise(lit("No"))
    )
    # Force execution to measure accurate time
    final_count = merged_df.count()
    
    # Force garbage collection
    gc.collect()
    
    # Track memory and time after conditional operations
    after_conditional_memory = get_memory_usage()
    conditional_memory_diff = after_conditional_memory - before_conditional_memory
    condition_time = time.time() - start_time
    print(f"Conditional operation completed in {condition_time:.4f} seconds.")
    print(f"Final dataframe has {final_count} rows.")
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
    merged_df.write.csv(f"{input_dir}/merged_data_spark", header=True, mode="overwrite")
    
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    perform_operations()
