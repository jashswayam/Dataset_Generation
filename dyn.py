import polars as pl
import re
from utils import DFoperations, ensure_list
import os

def parse_expression(expression, dataset_id=None):
    """
    Parse expressions like derived.mean, th.std_cycles, ds.column_name into appropriate polars expressions
    
    Parameters:
    - expression: string expression to parse
    - dataset_id: current dataset ID for ds. prefix resolution
    
    Returns:
    - Transformed expression ready for evaluation
    """
    def replace_match(match):
        prefix, field = match.groups()
        if prefix == "derived":
            return f"pl.col('derived_values').struct.field('{field}')"
        elif prefix == "th":
            return f"pl.col('static_thresholds').struct.field('{field}')"
        elif prefix == "dth":
            return f"pl.col('dynamic_thresholds').struct.field('{field}')"
        elif prefix == "ev":
            return f"pl.col('{field}')"
        elif prefix == "ds" and dataset_id:
            return f"pl.col('{field}')"
        return match.group(0)  # Return original if no match

    pattern = r"(\bderived|\bth|\bdth|\bev|\bds)\.(\w+)"
    transformed_expr = re.sub(pattern, replace_match, expression)
    return transformed_expr

def load_event_level_dfs(directory_path):
    """
    Load event level dataframes from parquet files
    
    Parameters:
    - directory_path: Path to directory containing parquet files
    
    Returns:
    - Dictionary mapping primary keys to their respective dataframes
    """
    event_level_dfs = {}
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.parquet'):
            filepath = os.path.join(directory_path, filename)
            df = pl.read_parquet(filepath)
            
            # Extract primary key from filename or from another source
            # This is an example, adjust according to your actual naming convention
            primary_key = filename.split('_')[0]  # Example: 'account_data.parquet' -> 'account'
            
            event_level_dfs[primary_key] = df
            
    return event_level_dfs

def dynamic_threshold(dynamic_thresholds: dict, datasets: dict, event_level_dfs: dict, lazy: bool = False):
    """
    Process dynamic thresholds calculations according to the new XML structure.
    
    Parameters:
    - dynamic_thresholds: The parsed XML containing the DynamicThresholdCalculations
    - datasets: Dictionary of dataframes referenced in the calculation
    - event_level_dfs: Dictionary of event level dataframes with their primary keys
    - lazy: Whether to use lazy evaluation
    
    Returns:
    - Updated event_level_dfs with dynamic threshold values
    """
    primary_key_info = dynamic_thresholds.get("PrimaryKey", {})
    primary_key_name = ensure_list(primary_key_info.get("Key", []))
    event_level_key = primary_key_info.get("EventLevelKey", primary_key_name[0] if primary_key_name else None)
    
    calculations = ensure_list(dynamic_thresholds.get("Calculation", []))
    
    # Get the appropriate event level dataframe
    if event_level_key not in event_level_dfs:
        raise KeyError(f"Event level dataframe with key '{event_level_key}' not found in provided dataframes")
    
    result_df = event_level_dfs[event_level_key].clone()
    
    # Create empty struct columns if they don't exist
    if "derived_values" not in result_df.columns:
        result_df = result_df.with_columns(pl.lit(None).alias("derived_values"))
    
    if "dynamic_thresholds" not in result_df.columns:
        result_df = result_df.with_columns(pl.lit(None).alias("dynamic_thresholds"))
    
    def get_polars_type(type_str):
        type_mapping = {
            'str': pl.Utf8,
            'float32': pl.Float32,
            'float64': pl.Float64,
            'int32': pl.Int32,
            'int64': pl.Int64,
            'bool': pl.Boolean
        }
        return type_mapping.get(type_str.lower(), pl.Utf8)
    
    # Process each calculation
    for calc in calculations:
        dataset_id = calc.get("DatasetId")
        if dataset_id not in datasets:
            print(f"Warning: Dataset {dataset_id} not found in datasets dictionary")
            continue
            
        join_key = ensure_list(calc.get("Keys", {}).get("Key", []))
        dataset_df = datasets[dataset_id].clone()
        
        if lazy:
            dataset_df = dataset_df.lazy()
        
        # Apply filters if any
        filters = ensure_list(calc.get("Filters", {}).get("Filter", []))
        if filters:
            # Update filter processing to handle ds. expressions
            processed_filters = []
            for filter_item in filters:
                column = filter_item.get("Column", "")
                operator = filter_item.get("Operator", "")
                value = filter_item.get("Value", "")
                
                # Parse column expression if it contains ds.
                if column.startswith("ds."):
                    column = column.replace("ds.", "", 1)
                
                # Parse value if it's an expression
                if isinstance(value, str) and ('.' in value):
                    value = parse_expression(value, dataset_id)
                    # Handle evaluation if needed
                    # (this may require more complex handling depending on your needs)
                
                processed_filters.append({
                    "Column": column,
                    "Operator": operator,
                    "Value": value
                })
            
            dataset_df = DFoperations.df_filters(dataset_df, processed_filters)
        
        # Process DerivedValues
        derived_values = calc.get("DerivedValues", {})
        column_name = derived_values.get("Column", "")
        
        # Check if GroupFunction exists for group by operation
        group_function = derived_values.get("GroupFunction")
        derived_values_dict = {}
        
        if group_function:
            # Split group functions if multiple (e.g., "mean, std")
            group_functions = [func.strip() for func in group_function.split(",")]
            
            # Get column and apply group by operation
            dataset_df = DFoperations.df_group_by(
                dataset_df, 
                join_key, 
                column_name,
                [column_name], 
                group_functions
            )
            
            # Each GroupFunction becomes a subindex name in derived_values
            for func in group_functions:
                func_col_name = f"{column_name}_{func}" if len(group_functions) > 1 else column_name
                derived_values_dict[func] = dataset_df[func_col_name]
        else:
            # No group by, just use the column as is
            derived_values_dict[column_name] = dataset_df[column_name]
        
        # Join the dataset to the event_level_df
        result_df = result_df.join(
            dataset_df.select(join_key + list(derived_values_dict.keys())),
            left_on=primary_key_name,
            right_on=join_key,
            how="left"
        )
        
        # Update derived_values struct with the new values
        for key, value in derived_values_dict.items():
            result_df = result_df.with_columns(
                pl.struct(["derived_values"]).update(pl.struct({key: pl.col(key)})).alias("derived_values")
            )
        
        # Process Values (upper_bound, lower_bound, etc.)
        values = ensure_list(calc.get("Values", {}).get("Value", []))
        for value in values:
            name = value.get("@name")
            val_type = value.get("@type")
            expression = value.get("#text", "")
            
            # Parse and evaluate the expression
            parsed_expr = parse_expression(expression, dataset_id)
            
            # Add the calculated value to the dynamic_thresholds struct
            try:
                result_df = result_df.with_columns(
                    pl.struct(["dynamic_thresholds"]).update(
                        pl.struct({name: eval(parsed_expr).cast(get_polars_type(val_type))})
                    ).alias("dynamic_thresholds")
                )
            except Exception as e:
                print(f"Error evaluating expression '{parsed_expr}' for value '{name}': {e}")
        
        # Drop temporary columns used for calculations
        cols_to_drop = list(derived_values_dict.keys())
        if cols_to_drop:
            result_df = result_df.drop(cols_to_drop)
    
    # Update the event level dataframe in the dictionary
    event_level_dfs[event_level_key] = result_df
    
    return event_level_dfs

# Example usage:
if __name__ == "__main__":
    import xmltodict
    import warnings
    warnings.filterwarnings("ignore")
    
    def get_memory_usage():
        """Returns the current memory usage in MB."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024
    
    # Load XML file
    with open('C:/Users/h59257/Downloads/rule_2.xml', 'r', encoding="utf-8") as file:
        xml_data = file.read()
    
    # Parse XML
    xml_dict = xmltodict.parse(xml_data)
    dynamic_thresholds = xml_dict.get("Rule").get("DynamicThresholdCalculations")
    
    # Load datasets from CSV files
    datasets = {"ds2": pl.read_csv("C:/Users/h59257/Downloads/profiles_sgp.csv")}
    
    # Load event level dataframes from parquet files
    event_level_directory = "C:/path/to/event_level_parquets"
    event_level_dfs = load_event_level_dfs(event_level_directory)
    
    # Alternatively, create a test event level dataframe
    if not event_level_dfs:
        event_level_dfs = {
            "account_id": pl.DataFrame({
                "account_id": ["1", "2", "3"],
                "static_thresholds": [
                    {"std_cycles": 2.5},
                    {"std_cycles": 3.0},
                    {"std_cycles": 1.8}
                ]
            })
        }
    
    # Track memory usage
    initial_memory = get_memory_usage()
    
    # Run dynamic threshold calculation
    updated_event_level_dfs = dynamic_threshold(dynamic_thresholds, datasets, event_level_dfs, lazy=False)
    
    # Final memory
    final_memory = get_memory_usage()
    
    # Print results
    for key, df in updated_event_level_dfs.items():
        print(f"\nEvent Level DF with key '{key}':")
        print(df)
    
    print(f"\nMemory usage: {final_memory - initial_memory} MB")