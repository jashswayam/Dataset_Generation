import pandas as pd
import polars as pl
import dask.dataframe as dd
import numpy as np
import time
import psutil
import gc
import os
import matplotlib.pyplot as plt
from functools import wraps

# Directory containing the datasets
DATA_DIR = "bank_data_joins"

# Performance tracking decorator
def measure_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clear memory before measurement
        gc.collect()
        process = psutil.Process()
        
        # Measure initial memory usage
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Measure final memory usage
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = end_mem - start_mem
        
        print(f"{func.__name__} - Time: {execution_time:.4f}s, Memory: {memory_usage:.2f} MB")
        
        return result, execution_time, memory_usage
    
    return wrapper

# Pandas operations
class PandasBenchmark:
    def __init__(self, data_dir, use_sample=False):
        self.data_dir = data_dir
        self.suffix = "_sample" if use_sample else ""
        print(f"Loading Pandas DataFrames from {data_dir}...")
        
    @measure_performance
    def load_data(self):
        transactions = pd.read_parquet(f"{self.data_dir}/transactions{self.suffix}.parquet")
        accounts = pd.rea
