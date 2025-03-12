from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import psutil
import time
import gc
import os

# Function to get current memory usage in MB
def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Memory in MB

def perform_operations(input_dir="bank_data_joins"):
    # Set Spark configurations to avoid truncation warning and improve performance
    os.environ['SPARK_LOCAL_DIRS'] = '/tmp'  # Ensure temp directory exists
    
    # Initialize Spark session with additional configurations
    spark = SparkSession.builder \
        .appName("BankDataProcessing") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.planChangeLog.level", "ERROR") \
        .config("spark.driver.extraJavaOptions", "-Dlog4j.rootCategory=WARN") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
    
    # Reduce logging verbosity
    spark.sparkContext.setLogLevel("ERROR")
    
    # Initial memory measurement
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Load the CSV data using PySpark
    print("\nLoading data...")
    load_start_time = time.time()
    
    # Force schema to avoid string type inference issues
    accounts_df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_dir}/accounts.csv")
    merchants_df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_dir}/merchants.csv")
    transactions_df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_dir}/transactions.csv")
    
    # Cache the DataFrames to improve performance
    accounts_df.cache()
    merchants_df.cache()
    transactions_df.cache()
    
    # Force execution to measure true loading time
    accounts_count = accounts_df.count()
    merchants_count = merchants_df.count()
    transactions_count = transactions_df.count()
    
    load_time = time.time() - load_start_time
    print(f"Data loading completed in {load_time:.4f} seconds.")
    print(f"Loaded {accounts_count} accounts, {merchants_count} merchants, {transactions_count} transactions.")
    
    # Memory after loading data
    gc.collect()  # Force garbage collection
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for loading data: {loading_memory_diff:.2f} MB")
    
    # ------ Filtering Operation ------
    print("\nPerforming filtering operation...")
    before_filtering_memory = get_memory_usage()
    filter_start_time = time.time()
    
    # Convert balance to double to ensure proper comparison
    filtered_accounts = accounts_df.filter(col("balance").cast("double") > 10000)
    # Cache and count to materialize
    filtered_accounts.cache()
    filtered_count = filtered_accounts.count()
    
    filter_time = time.time() - filter_start_time
    print(f"Filtering operation completed in {filter_time:.4f} seconds.")
    print(f"Filtered to {filtered_count} accounts with balance > 10000.")
    
    # Force garbage collection and measure memory
    gc.collect()
    after_filtering_memory = get_memory_usage()
    filtering_memory_diff = after_filtering_memory - before_filtering_memory
    print(f"Memory used by filtering operation: {filtering_memory_diff:.2f} MB")
    
    # ------ Merging Operation ------
    print("\nPerforming merging operation...")
    before_merging_memory = get_memory_usage()
    merge_start_time = time.time()
    
    # Perform the join operation (inner join on account_id and merchant_id)
    merged_df = filtered_accounts.join(transactions_df, on="account_id", how="inner")
    merged_df = merged_df.join(merchants_df, on="merchant_id", how="inner")
    
    # Cache and count to materialize
    merged_df.cache()
    merged_count = merged_df.count()
    
    merge_time = time.time() - merge_start_time
    print(f"Merging operation completed in {merge_time:.4f} seconds.")
    print(f"Merged data contains {merged_count} rows.")
    
    # Force garbage collection and measure memory
    gc.collect()
    after_merging_memory = get_memory_usage()
    merging_memory_diff = after_merging_memory - before_merging_memory
    print(f"Memory used by merging operation: {merging_memory_diff:.2f} MB")
    
    # ------ Conditional Operation ------
    print("\nPerforming conditional operation...")
    before_conditional_memory = get_memory_usage()
    condition_start_time = time.time()
    
    # Add a new column for high_value_transaction
    # Convert amount to double to ensure proper comparison
    merged_df = merged_df.withColumn(
        "high_value_transaction", 
        when(col("amount").cast("double") > 500, "Yes").otherwise("No")
    )
    
    # Cache and count to materialize
    merged_df.cache()
    final_count = merged_df.count()
    high_value_count = merged_df.filter(col("high_value_transaction") == "Yes").count()
    
    condition_time = time.time() - condition_start_time
    print(f"Conditional operation completed in {condition_time:.4f} seconds.")
    print(f"Found {high_value_count} high-value transactions out of {final_count} total.")
    
    # Force garbage collection and measure memory
    gc.collect()
    after_conditional_memory = get_memory_usage()
    conditional_memory_diff = after_conditional_memory - before_conditional_memory
    print(f"Memory used by conditional operation: {conditional_memory_diff:.2f} MB")
    
    # ------ Generate a report ------
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print(f"Memory used for loading data: {loading_memory_diff:.2f} MB")
    print(f"Memory used by filtering operation: {filtering_memory_diff:.2f} MB")
    print(f"Memory used by merging operation: {merging_memory_diff:.2f} MB")
    print(f"Memory used by conditional operation: {conditional_memory_diff:.2f} MB")
    print(f"Total memory increase: {after_conditional_memory - initial_memory:.2f} MB")
    
    print(f"\nData loading took: {load_time:.4f} seconds.")
    print(f"Filtering operation took: {filter_time:.4f} seconds.")
    print(f"Merging operation took: {merge_time:.4f} seconds.")
    print(f"Conditional operation took: {condition_time:.4f} seconds.")
    print(f"Total processing time: {load_time + filter_time + merge_time + condition_time:.4f} seconds.")
    
    # Optionally save the final result to a new CSV file
    print("\nSaving results...")
    save_start_time = time.time()
    
    # Use coalesce to reduce the number of output files
    merged_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{input_dir}/merged_data_spark")
    
    save_time = time.time() - save_start_time
    print(f"Results saved in {save_time:.4f} seconds.")
    
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    perform_operations()
