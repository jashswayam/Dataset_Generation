import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import pyarrow as pa
import pyarrow.parquet as pq

def generate_bank_transactions(num_rows=10_000_000, output_dir="bank_data", chunk_size=1_000_000):
    """
    Generate a large dataset of bank transactions for benchmarking.
    
    Parameters:
    -----------
    num_rows : int
        Number of transaction records to generate
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
    
    print(f"Generating {num_rows:,} bank transactions...")
    
    # Generate data in chunks to manage memory
    num_chunks = (num_rows + chunk_size - 1) // chunk_size
    
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = min((chunk + 1) * chunk_size, num_rows)
        current_size = end_idx - start_idx
        
        print(f"Generating chunk {chunk+1}/{num_chunks} ({current_size:,} records)...")
        
        # Generate random data
        random_days = np.random.randint(0, date_range, size=current_size)
        dates = [base_date + timedelta(days=int(d)) for d in random_days]
        
        account_indices = np.random.randint(0, len(account_ids), size=current_size)
        account_id = [account_ids[i] for i in account_indices]
        
        transaction_id = [f"TXN{str(start_idx+i).zfill(10)}" for i in range(current_size)]
        
        # Generate continuous amounts between -5000 and 10000 with more values clustered around common amounts
        amounts = np.random.exponential(scale=100, size=current_size)
        # Add some negative values for withdrawals
        negative_mask = np.random.random(size=current_size) < 0.4
        amounts[negative_mask] = -amounts[negative_mask]
        # Add some larger values (deposits/withdrawals)
        large_mask = np.random.random(size=current_size) < 0.05
        amounts[large_mask] = amounts[large_mask] * 10
        # Clip to reasonable range and round to 2 decimal places
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
    
    # Create a small sample file for quick testing
    sample_size = min(10_000, num_rows)
    sample_df = pd.read_parquet(f"{output_dir}/transactions_chunk_1.parquet").iloc[:sample_size]
    sample_df.to_parquet(f"{output_dir}/transactions_sample.parquet")
    sample_df.to_csv(f"{output_dir}/transactions_sample.csv", index=False)
    print(f"Created sample files with {sample_size:,} records for quick testing")

if __name__ == "__main__":
    # You can adjust these parameters based on your system's capabilities
    generate_bank_transactions(
        num_rows=10_000_000,  # 10 million rows
        output_dir="bank_data",
        chunk_size=1_000_000  # 1 million rows per file
    )
