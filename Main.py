import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import pyarrow as pa
import pyarrow.parquet as pq

def generate_bank_data_for_joins(output_dir="bank_data_joins"):
    """
    Generate three separate files of 1 million entries each for join operations:
    1. transactions.parquet - core transaction data
    2. accounts.parquet - account information
    3. merchants.parquet - merchant information
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Generating bank data for join operations...")
    
    # ------ Generate accounts data (1 million entries) ------
    print("Generating accounts data (1M entries)...")
    
    # Generate account IDs
    num_accounts = 1_000_000
    account_ids = [f"ACCT{str(i).zfill(8)}" for i in range(num_accounts)]
    
    # Generate account types with realistic distributions
    account_types = ['Checking', 'Savings', 'Credit Card', 'Investment', 'Loan']
    account_type_dist = np.random.choice(account_types, size=num_accounts, p=[0.4, 0.3, 0.2, 0.05, 0.05])
    
    # Generate customer IDs (some accounts belong to the same customer)
    num_customers = 500_000
    customer_ids = [f"CUST{str(i).zfill(8)}" for i in range(num_customers)]
    account_to_customer = np.random.choice(customer_ids, size=num_accounts)
    
    # Generate account open dates
    base_date = datetime(2010, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - base_date).days
    random_days = np.random.randint(0, date_range, size=num_accounts)
    open_dates = [base_date + timedelta(days=int(d)) for d in random_days]
    
    # Generate account balances
    balances = np.random.exponential(scale=5000, size=num_accounts)
    large_balance_mask = np.random.random(size=num_accounts) < 0.1
    balances[large_balance_mask] = balances[large_balance_mask] * 10
    balances = np.round(balances, 2)
    
    # Generate account status
    status_options = ['Active', 'Inactive', 'Closed', 'Suspended']
    status = np.random.choice(status_options, size=num_accounts, p=[0.85, 0.1, 0.04, 0.01])
    
    # Create accounts DataFrame
    accounts_df = pd.DataFrame({
        'account_id': account_ids,
        'customer_id': account_to_customer,
        'account_type': account_type_dist,
        'open_date': open_dates,
        'balance': balances,
        'status': status,
        'credit_score': np.random.randint(300, 851, size=num_accounts),
        'overdraft_limit': np.random.choice([0, 100, 500, 1000, 2000], size=num_accounts),
        'interest_rate': np.round(np.random.uniform(0.01, 0.1, size=num_accounts), 4)
    })
    
    # Save accounts data
    accounts_filename = f"{output_dir}/accounts.parquet"
    accounts_table = pa.Table.from_pandas(accounts_df)
    pq.write_table(accounts_table, accounts_filename)
    accounts_df.to_csv(f"{output_dir}/accounts.csv", index=False)
    print(f"Saved accounts data to {accounts_filename}")
    
    # ------ Generate merchants data (1 million entries) ------
    print("Generating merchants data (1M entries)...")
    
    # Generate merchant IDs
    num_merchants = 1_000_000
    merchant_ids = [f"MERCH{str(i).zfill(8)}" for i in range(num_merchants)]
    
    # Generate merchant categories
    categories = [
        'Groceries', 'Restaurant', 'Fast Food', 'Entertainment', 'Shopping', 
        'Clothing', 'Electronics', 'Transportation', 'Travel', 'Hotels', 
        'Utilities', 'Housing', 'Healthcare', 'Education', 'Professional Services',
        'Financial Services', 'Insurance', 'Automotive', 'Home Improvement', 'Pets'
    ]
    
    # Generate merchant names (some real, some synthetic)
    real_merchants = [
        'Amazon', 'Walmart', 'Target', 'Costco', 'Uber', 'Lyft', 'Netflix', 'Spotify',
        'Apple', 'Google', 'Microsoft', 'Tesla', 'Shell', 'BP', 'Exxon', 'Chevron',
        'Starbucks', 'McDonalds', 'Subway', 'Chipotle', 'Home Depot', 'Lowes',
        'Best Buy', 'Whole Foods', 'Trader Joes', 'CVS', 'Walgreens', 'Kroger',
        'Safeway', 'Publix', 'Aldi', 'Lidl', 'T-Mobile', 'Verizon', 'AT&T', 'Comcast',
        'Delta', 'United', 'American Airlines', 'Southwest', 'Marriott', 'Hilton', 'Hyatt'
    ]
    
    # Generate merchant names (mix of real and synthetic)
    merchant_names = []
    for i in range(num_merchants):
        if i < len(real_merchants):
            merchant_names.append(real_merchants[i])
        else:
            prefix = random.choice(['Super', 'Mega', 'Ultra', 'Best', 'Prime', 'Quality', 'Value', 'Premium', 'Discount', ''])
            suffix = random.choice(['Store', 'Market', 'Shop', 'Outlet', 'Place', 'Center', 'Services', 'Solutions', ''])
            if prefix == '' and suffix == '':
                suffix = 'Inc'
            
            base_name = random.choice([
                'Food', 'Mart', 'Tech', 'Health', 'Auto', 'Home', 'Garden', 'Pet', 'Sport', 'Craft',
                'Beauty', 'Fashion', 'Repair', 'Clean', 'Build', 'Fix', 'Go', 'Quick', 'Express', 'Global'
            ])
            
            merchant_names.append(f"{prefix} {base_name} {suffix}".strip())
    
    # Generate merchant categories
    merchant_category = np.random.choice(categories, size=num_merchants)
    
    # Generate merchant locations (country codes)
    countries = ['US', 'CA', 'UK', 'DE', 'FR', 'JP', 'CN', 'AU', 'MX', 'BR']
    country_codes = np.random.choice(countries, size=num_merchants, p=[0.7, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02])
    
    # Generate merchant ratings
    ratings = np.random.uniform(1.0, 5.0, size=num_merchants)
    ratings = np.round(ratings * 2) / 2  # Round to nearest 0.5
    
    # Create merchants DataFrame
    merchants_df = pd.DataFrame({
        'merchant_id': merchant_ids,
        'merchant_name': merchant_names,
        'category': merchant_category,
        'country': country_codes,
        'rating': ratings,
        'is_online': np.random.random(size=num_merchants) < 0.4,
        'transaction_count': np.random.randint(10, 100000, size=num_merchants),
        'average_transaction': np.round(np.random.exponential(scale=50, size=num_merchants), 2)
    })
    
    # Save merchants data
    merchants_filename = f"{output_dir}/merchants.parquet"
    merchants_table = pa.Table.from_pandas(merchants_df)
    pq.write_table(merchants_table, merchants_filename)
    merchants_df.to_csv(f"{output_dir}/merchants.csv", index=False)
    print(f"Saved merchants data to {merchants_filename}")
    
    # ------ Generate transactions data (1 million entries) ------
    print("Generating transactions data (1M entries)...")
    
    num_transactions = 1_000_000
    transaction_ids = [f"TXN{str(i).zfill(10)}" for i in range(num_transactions)]
    
    # Use existing account IDs and merchant IDs for joining
    transaction_account_ids = np.random.choice(account_ids, size=num_transactions)
    transaction_merchant_ids = np.random.choice(merchant_ids, size=num_transactions)
    
    # Generate transaction dates
    base_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - base_date).days
    random_days = np.random.randint(0, date_range, size=num_transactions)
    transaction_dates = [base_date + timedelta(days=int(d)) for d in random_days]
    
    # Generate transaction types
    transaction_types = ['Purchase', 'Withdrawal', 'Deposit', 'Transfer', 'Payment', 'Fee', 'Interest', 'Refund']
    transaction_type_values = np.random.choice(transaction_types, size=num_transactions)
    
    # Generate amounts
    amounts = np.random.exponential(scale=100, size=num_transactions)
    negative_mask = np.random.random(size=num_transactions) < 0.4
    amounts[negative_mask] = -amounts[negative_mask]
    large_mask = np.random.random(size=num_transactions) < 0.05
    amounts[large_mask] = amounts[large_mask] * 10
    amounts = np.round(amounts, 2)
    
    # Generate transaction status
    status_options = ['Completed', 'Pending', 'Failed', 'Disputed', 'Refunded']
    status = np.random.choice(status_options, size=num_transactions, p=[0.94, 0.03, 0.01, 0.01, 0.01])
    
    # Create transactions DataFrame
    transactions_df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'account_id': transaction_account_ids,
        'merchant_id': transaction_merchant_ids,
        'date': transaction_dates,
        'amount': amounts,
        'transaction_type': transaction_type_values,
        'status': status,
        'is_online': np.random.random(size=num_transactions) < 0.7,
        'is_recurring': np.random.random(size=num_transactions) < 0.25,
        'processing_fee': np.round(np.random.uniform(0, 5, size=num_transactions), 2)
    })
    
    # Save transactions data
    transactions_filename = f"{output_dir}/transactions.parquet"
    transactions_table = pa.Table.from_pandas(transactions_df)
    pq.write_table(transactions_table, transactions_filename)
    transactions_df.to_csv(f"{output_dir}/transactions.csv", index=False)
    print(f"Saved transactions data to {transactions_filename}")
    
    # Create sample files for quick testing
    sample_size = 10_000
    accounts_df.iloc[:sample_size].to_parquet(f"{output_dir}/accounts_sample.parquet")
    merchants_df.iloc[:sample_size].to_parquet(f"{output_dir}/merchants_sample.parquet")
    transactions_df.iloc[:sample_size].to_parquet(f"{output_dir}/transactions_sample.parquet")
    
    print(f"\nGenerated 3 files with 1M entries each in {output_dir}/")
    print("Files can be joined on:")
    print("- transactions.account_id = accounts.account_id")
    print("- transactions.merchant_id = merchants.merchant_id")
    print("Sample files with 10K entries each were also created for quick testing")

if __name__ == "__main__":
    generate_bank_data_for_joins(output_dir="bank_data_joins")
