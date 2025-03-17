import dask.dataframe as dd
import numpy as np
import pandas as pd
import psutil
import time
import gc
import numba
from numba import jit, prange, int32, float32
from dask.diagnostics import ProgressBar
from dask import delayed

def get_memory_usage():
    """
    Returns the current memory usage in MB.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Memory in MB

# Define optimized data types
DTYPE_MAP = {
    'account_id': np.int32,
    'merchant_id': np.int32,
    'transaction_id': np.int32,
    'amount': np.float32,
    'balance': np.float32
}

# Define Numba-accelerated functions with explicit types
@jit(float32(float32[:]), nopython=True, parallel=True, fastmath=True)
def numba_sum_float32(arr):
    result = 0.0
    for i in prange(len(arr)):
        result += arr[i]
    return result

@jit(float32(float32[:]), nopython=True, parallel=True, fastmath=True)
def numba_mean_float32(arr):
    if len(arr) == 0:
        return 0.0
    return numba_sum_float32(arr) / len(arr)

@jit(int32(int32[:]), nopython=True, parallel=True)
def numba_unique_count_int32(arr):
    # Sort the array (faster than set for large arrays)
    sorted_arr = np.sort(arr)
    
    # Count unique elements
    if len(sorted_arr) == 0:
        return 0
        
    count = 1
    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] != sorted_arr[i-1]:
            count += 1
    
    return count

# Compact preprocessing function
@jit(nopython=True)
def preprocess_partition(transaction_ids, account_ids, merchant_ids, amounts, balances, categories, statuses):
    """
    Preprocesses a partition of data using Numba
    Returns filtered arrays and a high_value flag array
    """
    n = len(transaction_ids)
    # Create active status mask
    active_mask = np.zeros(n, dtype=numba.boolean)
    for i in range(n):
        if statuses[i] == 'Active':
            active_mask[i] = True
    
    # Count active records
    active_count = np.sum(active_mask)
    
    # Initialize result arrays
    result_transaction_ids = np.empty(active_count, dtype=np.int32)
    result_account_ids = np.empty(active_count, dtype=np.int32)
    result_merchant_ids = np.empty(active_count, dtype=np.int32)
    result_amounts = np.empty(active_count, dtype=np.float32)
    result_balances = np.empty(active_count, dtype=np.float32)
    result_categories = np.empty(active_count, dtype=numba.types.string)
    result_high_value = np.empty(active_count, dtype=numba.types.string)
    
    # Fill result arrays
    idx = 0
    for i in range(n):
        if active_mask[i]:
            result_transaction_ids[idx] = transaction_ids[i]
            result_account_ids[idx] = account_ids[i]
            result_merchant_ids[idx] = merchant_ids[i]
            result_amounts[idx] = amounts[i]
            result_balances[idx] = balances[i]
            result_categories[idx] = categories[i]
            # Set high_value flag
            if amounts[i] > 500.0:
                result_high_value[idx] = 'Yes'
            else:
                result_high_value[idx] = 'No'
            idx += 1
    
    return (result_transaction_ids, result_account_ids, result_merchant_ids, 
            result_amounts, result_balances, result_categories, result_high_value)

# Ultra-fast groupby implementation
@jit(nopython=True)
def fast_groupby(categories, high_value_flags, amounts, balances, account_ids, merchant_ids):
    """
    Performs a groupby operation using Numba with minimal memory usage
    Returns category, high_value, sum, mean, count, unique_accounts, unique_merchants
    """
    # Find unique combinations of category and high_value_flag
    unique_pairs = set()
    for i in range(len(categories)):
        unique_pairs.add((categories[i], high_value_flags[i]))
    
    # Convert to list for iteration
    unique_pairs_list = list(unique_pairs)
    
    # For each unique combination, compute aggregations
    results = []
    for cat, hv in unique_pairs_list:
        # Create mask for this group
        mask = np.zeros(len(categories), dtype=numba.boolean)
        for i in range(len(categories)):
            if categories[i] == cat and high_value_flags[i] == hv:
                mask[i] = True
        
        # Get masked arrays
        masked_amounts = amounts[mask]
        masked_balances = balances[mask]
        masked_account_ids = account_ids[mask]
        masked_merchant_ids = merchant_ids[mask]
        
        # Compute aggregations
        amount_sum = numba_sum_float32(masked_amounts)
        amount_mean = numba_mean_float32(masked_amounts)
        amount_count = len(masked_amounts)
        balance_mean = numba_mean_float32(masked_balances)
        unique_accounts = numba_unique_count_int32(masked_account_ids)
        unique_merchants = numba_unique_count_int32(masked_merchant_ids)
        
        # Add result
        results.append((cat, hv, amount_sum, amount_mean, amount_count, 
                        balance_mean, unique_accounts, unique_merchants))
    
    return results

def perform_operations(input_dir="bank_data_joins"):
    # Initial memory measurement
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Load the generated data with optimized data types
    print("\nLoading data with optimized types...")
    
    # Define dtype for faster loading
    dtypes = {
        'account_id': 'int32',
        'merchant_id': 'int32',
        'transaction_id': 'int32',
        'amount': 'float32',
        'balance': 'float32'
    }
    
    # Load only needed columns with optimized types
    accounts_df = dd.read_parquet(
        f"{input_dir}/accounts.parquet",
        engine='pyarrow',
        columns=['account_id', 'balance', 'status'],
        dtype={
            'account_id': 'int32',
            'balance': 'float32'
        }
    )
    
    merchants_df = dd.read_parquet(
        f"{input_dir}/merchants.parquet",
        engine='pyarrow',
        columns=['merchant_id', 'category'],
        dtype={
            'merchant_id': 'int32'
        }
    )
    
    transactions_df = dd.read_parquet(
        f"{input_dir}/transactions.parquet",
        engine='pyarrow',
        columns=['transaction_id', 'account_id', 'merchant_id', 'amount'],
        dtype={
            'transaction_id': 'int32',
            'account_id': 'int32',
            'merchant_id': 'int32',
            'amount': 'float32'
        }
    )
    
    # Memory after loading
    loading_memory = get_memory_usage()
    loading_memory_diff = loading_memory - initial_memory
    print(f"Memory used for loading: {loading_memory_diff:.2f} MB")
    
    # ------ Ultra-optimized pipeline ------
    print("\nPerforming ultra-optimized pipeline...")
    start_pipeline_time = time.time()
    
    # Process in batches for memory efficiency
    all_results = []
    
    # Set up progress bar
    with ProgressBar():
        # Precompute merchants (usually small)
        merchants_dict = merchants_df.compute().set_index('merchant_id')['category'].to_dict()
        
        # Process each transaction partition
        print(f"Processing {transactions_df.npartitions} partitions...")
        for i in range(transactions_df.npartitions):
            @delayed
            def process_partition(i):
                print(f"Processing partition {i}...")
                # Get transaction partition
                txn_part = transactions_df.get_partition(i).compute()
                
                # Convert to numpy arrays with optimized types
                transaction_ids = txn_part['transaction_id'].to_numpy(dtype=np.int32)
                account_ids = txn_part['account_id'].to_numpy(dtype=np.int32)
                merchant_ids = txn_part['merchant_id'].to_numpy(dtype=np.int32)
                amounts = txn_part['amount'].to_numpy(dtype=np.float32)
                
                # Get account data for this partition
                unique_account_ids = np.unique(account_ids)
                accounts_part = accounts_df[accounts_df['account_id'].isin(unique_account_ids)].compute()
                
                # Create account lookup dictionaries
                balance_dict = accounts_part.set_index('account_id')['balance'].to_dict()
                status_dict = accounts_part.set_index('account_id')['status'].to_dict()
                
                # Create arrays for accounts and merchants
                balances = np.zeros(len(account_ids), dtype=np.float32)
                statuses = np.empty(len(account_ids), dtype=object)
                categories = np.empty(len(merchant_ids), dtype=object)
                
                # Fill arrays using dictionaries
                for j in range(len(account_ids)):
                    acc_id = account_ids[j]
                    merch_id = merchant_ids[j]
                    
                    # Set balance and status
                    if acc_id in balance_dict:
                        balances[j] = balance_dict[acc_id]
                        statuses[j] = status_dict[acc_id]
                    else:
                        balances[j] = 0.0
                        statuses[j] = 'Inactive'
                    
                    # Set category
                    if merch_id in merchants_dict:
                        categories[j] = merchants_dict[merch_id]
                    else:
                        categories[j] = 'Unknown'
                
                # Preprocess data
                (filtered_transaction_ids, filtered_account_ids, filtered_merchant_ids,
                 filtered_amounts, filtered_balances, filtered_categories, 
                 high_value_flags) = preprocess_partition(
                     transaction_ids, account_ids, merchant_ids, amounts,
                     balances, categories, statuses
                 )
                
                # Perform groupby if we have data
                if len(filtered_transaction_ids) > 0:
                    # Calculate groupby results
                    results = fast_groupby(
                        filtered_categories, high_value_flags, filtered_amounts,
                        filtered_balances, filtered_account_ids, filtered_merchant_ids
                    )
                    
                    # Convert results to DataFrame
                    result_df = pd.DataFrame(results, columns=[
                        'category', 'high_value_transaction', 'amount_sum', 'amount_mean',
                        'amount_count', 'balance_mean', 'unique_accounts', 'unique_merchants'
                    ])
                    
                    return result_df
                else:
                    # Return empty DataFrame with correct columns
                    return pd.DataFrame(columns=[
                        'category', 'high_value_transaction', 'amount_sum', 'amount_mean',
                        'amount_count', 'balance_mean', 'unique_accounts', 'unique_merchants'
                    ])
            
            # Add partition result to list
            all_results.append(process_partition(i))
        
        # Combine results from all partitions
        @delayed
        def combine_results(dfs):
            # Concatenate all DataFrames
            combined = pd.concat(dfs, ignore_index=True)
            
            # Perform final aggregation
            if len(combined) > 0:
                final = combined.groupby(['category', 'high_value_transaction']).agg({
                    'amount_sum': 'sum',
                    'amount_mean': 'mean',
                    'amount_count': 'sum',
                    'balance_mean': 'mean',
                    'unique_accounts': 'sum',
                    'unique_merchants': 'sum'
                }).reset_index()
                
                return final
            else:
                return combined
        
        # Combine all results
        print("Combining results from all partitions...")
        final_result_delayed = combine_results(all_results)
        
        # Compute final result
        print("Computing final result...")
        collection_start_time = time.time()
        result = final_result_delayed.compute()
        collection_time = time.time() - collection_start_time
    
    # End of pipeline
    pipeline_time = time.time() - start_pipeline_time
    print(f"Pipeline completed in {pipeline_time:.4f} seconds.")
    print(f"Final collection time: {collection_time:.4f} seconds.")
    
    # Display result
    print("\nSample of the result:")
    print(result.head(5))
    
    # Force garbage collection
    gc.collect()
    
    # Track memory
    final_memory = get_memory_usage()
    memory_diff = final_memory - initial_memory
    print(f"Total memory increase: {memory_diff:.2f} MB")
    
    # ------ Generate a summary report ------
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    print(f"Memory used for loading: {loading_memory_diff:.2f} MB")
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Total memory increase: {memory_diff:.2f} MB")
    
    print(f"\nTotal pipeline time: {pipeline_time:.4f} seconds.")
    print(f"Collection time: {collection_time:.4f} seconds.")
    
    # Save results
    result.to_parquet(f"{input_dir}/grouped_data_optimized.parquet", compression='snappy')
    result.to_csv(f"{input_dir}/grouped_data_optimized.csv")
    
    return result

if __name__ == "__main__":
    perform_operations()