from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import psutil
import time

# Function to get current memory usage in MB
def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Memory in MB

def perform_operations(input_dir="bank_data_joins"):
    # Initialize Spark session
    spark = SparkSession.builder.appName("BankDataProcessing").getOrCreate()
    
    # Load the CSV data using PySpark
    print("Loading data...")
    accounts_df = spark.read.option("header", "true").csv(f"{input_dir}/accounts.csv")
    merchants_df = spark.read.option("header", "true").csv(f"{input_dir}/merchants.csv")
    transactions_df = spark.read.option("header", "true").csv(f"{input_dir}/transactions.csv")
    
    # Track memory usage before operations
    mem_before = get_memory_usage()
    print(f"Memory before operation: {mem_before:.2f} MB")

    # Track the time before filtering operation
    start_time = time.time()

    # ------ Filtering Operation ------
    print("Performing filtering operation...")
    filtered_accounts = accounts_df.filter(col("balance") > 10000)  # Example: filter accounts with balance > 10K
    
    # Time taken for filtering
    filter_time = time.time() - start_time
    print(f"Filtering operation completed in {filter_time:.4f} seconds.")

    # Track memory usage after filtering
    mem_after_filter = get_memory_usage()
    print(f"Memory after filtering operation: {mem_after_filter:.2f} MB")

    # ------ Merging Operation ------
    start_time = time.time()

    print("Performing merging operation...")
    # Perform the join operation (inner join on account_id and merchant_id)
    merged_df = filtered_accounts.join(transactions_df, on="account_id", how="inner") \
                                 .join(merchants_df, on="merchant_id", how="inner")

    # Time taken for merging
    merge_time = time.time() - start_time
    print(f"Merging operation completed in {merge_time:.4f} seconds.")

    # Track memory usage after merging
    mem_after_merge = get_memory_usage()
    print(f"Memory after merging operation: {mem_after_merge:.2f} MB")

    # ------ Conditional Operation ------
    start_time = time.time()

    print("Performing conditional operation...")
    # Add a new column for high_value_transaction
    merged_df = merged_df.withColumn(
        "high_value_transaction", 
        when(col("amount") > 500, "Yes").otherwise("No")
    )

    # Time taken for conditional operation
    condition_time = time.time() - start_time
    print(f"Conditional operation completed in {condition_time:.4f} seconds.")

    # Track memory usage after conditional operation
    mem_after_condition = get_memory_usage()
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
    merged_df.write.option("header", "true").csv(f"{input_dir}/merged_data.csv")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    perform_operations()
