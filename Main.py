To ensure that the generated data in the three chunks can be used for performing join operations, we'll need to introduce some common fields between the chunks, such as account_id, which can act as the primary key for joins. Here’s how you can adjust your code to ensure that the account_id values are consistent across the three files, and that you generate only three files with 1 million entries each.

Key Changes:

1. Consistent Account IDs Across Chunks: We need to ensure that each chunk contains some overlapping account_id values to enable the join operation.


2. Reduce Number of Chunks: We'll modify the number of chunks to 3, each containing 1 million rows.


3. Ensure Joinability: We'll include an account_id column that will be consistent across chunks, allowing them to be joined.



Modified Code:

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import pyarrow as pa
import pyarrow.parquet as pq

def generate_bank_transactions(num_rows=3_000_000, output_dir="bank_data", chunk_size=1_000_000):
    """
    Generate a dataset of bank transactions for benchmarking with joinable chunks.
    
    Parameters:
    -----------
    num_rows : int
        Total number of transaction records to generate across chunks.
    output_dir : str
        Directory to save the output files
    chunk_size : int
        Number of rows per file chunk
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Constants for data generation
    account_types = ['Checking', 'Savings', 'Credit Card', 'Investment', 'Loan']
    transaction_types = ['Deposit', 'Withdrawal', 'Transfer', 'Payment', 'Fee', 'Interest', 'Refund']
    merchants = [
        'Amazon', 'Walmart', 'Target', 'Costco', 'Uber', 'Lyft', 'Netflix', 'Spotify',
        'Apple', 'Google', 'Microsoft', 'Tesla', 'Shell', 'BP', 'Exxon', 'Chevron',
        'Starbucks', 'McDonalds', 'Subway', 'Chipotle', 'Home Depot', 'Lowes',
        'Best Buy', 'Whole Foods', 'Trader Joes', 'CVS', 'Walgreens'
    ]
    categories = [
        'Groceries', 'Restaurant', 'Entertainment', 'Shopping', 'Transportation',
        'Travel', 'Utilities', 'Housing', 'Healthcare', 'Education', 'Income'
    ]
    status_options = ['Completed', 'Pending', 'Failed', 'Disputed', 'Refunded']
    
    # Start with a base date and generate random dates within a range
    base_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - base_date).days
    
    # Generate account IDs (10,000 unique accounts)
    num_accounts = 10_000
    account_ids = [f"ACCT{str(i).zfill(8)}" for i in range(num_accounts)]
    
    print(f"Generating {num_rows:,} bank transactions across {num_rows//chunk_size} chunks...")
    
    # Generate data in chunks to manage memory and ensure joins
    num_chunks = (num_rows + chunk_size - 1) // chunk_size  # Ensure we generate 3 chunks
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = min((chunk + 1) * chunk_size, num_rows)
        current_size = end_idx - start_idx
        
        print(f"Generating chunk {chunk+1}/{num_chunks} ({current_size:,} records)...")
        
        # Generate random data
        random_days = np.random.randint(0, date_range, size=current_size)
        dates = [base_date + timedelta(days=int(d)) for d in random_days]
        
        # Select a subset of account_ids for this chunk, ensuring overlap with previous chunks
        # In this case, we'll overlap the account IDs by selecting them from a pre-shuffled list
        account_indices = np.random.randint(0, len(account_ids), size=current_size)
        account_id = [account_ids[i] for i in account_indices]
        
        transaction_id = [f"TXN{str(start_idx+i).zfill(10)}" for i in range(current_size)]
        
        # Generate continuous amounts between -5000 and 10000 with more values clustered around common amounts
        amounts = np.random.exponential(scale=100, size=current_size)
        negative_mask = np.random.random(size=current_size) < 0.4
        amounts[negative_mask] = -amounts[negative_mask]
        large_mask = np.random.random(size=current_size) < 0.05
        amounts[large_mask] = amounts[large_mask] * 10
        amounts = np.clip(amounts, -5000, 10000)
        amounts = np.round(amounts, 2)
        
        # Generate other categorical fields with realistic distributions
        account_type = np.random.choice(account_types, size=current_size, p=[0.4, 0.3, 0.2, 0.05, 0.05])
        transaction_type = np.random.choice(transaction_types, size=current_size)
        merchant = np.random.choice(merchants + [None], size=current_size, p=[0.025]*len(merchants) + [0.325])
        category = np.random.choice(categories, size=current_size)
        status = np.random.choice(status_options, size=current_size, p=[0.94, 0.03, 0.01, 0.01, 0.01])
        
        # Generate running balances
        balances = np.cumsum(amounts) + 5000  # Starting with a base balance of 5000
        balances = np.round(balances, 2)
        
        # Generate boolean flags
        is_online = np.random.random(size=current_size) < 0.7
        is_recurring = np.random.random(size=current_size) < 0.25
        
        # Create a DataFrame
        df = pd.DataFrame({
            'transaction_id': transaction_id,
            'account_id': account_id,
            'date': dates,
            'amount': amounts,
            'balance': balances,
            'transaction_type': transaction_type,
            'account_type': account_type,
            'merchant': merchant,
            'category': category,
            'status': status,
            'is_online': is_online,
            'is_recurring': is_recurring
        })
        
        # Save chunk to files in both formats
        parquet_filename = f"{output_dir}/transactions_chunk_{chunk+1}.parquet"
        csv_filename = f"{output_dir}/transactions_chunk_{chunk+1}.csv"
        
        # Save as Arrow/Parquet file (more efficient)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_filename)
        
        # Also save a CSV for libraries that prefer it
        df.to_csv(csv_filename, index=False)
        
        print(f"Saved chunk to {parquet_filename} and {csv_filename}")
    
    print(f"\nGenerated {num_rows:,} bank transactions in {output_dir}/")

if __name__ == "__main__":
    # You can adjust these parameters based on your system's capabilities
    generate_bank_transactions(
        num_rows=3_000_000,  # 3 million rows total, split into 3 chunks
        output_dir="bank_data",
        chunk_size=1_000_000  # 1 million rows per file
    )
