import polars as pl
import re
from utils import DFoperations, ensure_list

def parse_expression(expression):
    """Parse expressions like derived.mean or th.std_cycles into appropriate polars expressions"""
    def replace_match(match):
        prefix, field = match.groups()
        if prefix == "derived":
            return f"pl.col('derived').struct.field('{field}')"
        elif prefix == "th":
            return f"pl.col('static_thresholds').struct.field('{field}')"
        elif prefix == "ev":
            return f"pl.col('{field}')"
        return match.group(0)  # Return original if no match

    pattern = r"(\bderived|\bth|\bev)\.(\w+)"
    transformed_expr = re.sub(pattern, replace_match, expression)
    return transformed_expr

def dynamic_threshold(dynamic_thresholds: dict, datasets: dict, event_level_df: pl.DataFrame, lazy: bool = False):
    """
    Process dynamic thresholds calculations according to the new XML structure.
    
    Parameters:
    - dynamic_thresholds: The parsed XML containing the DynamicThresholdCalculations
    - datasets: Dictionary of dataframes referenced in the calculation
    - event_level_df: The event level dataframe which will receive the calculated values
    - lazy: Whether to use lazy evaluation
    
    Returns:
    - Updated event_level_df with dynamic threshold values
    """
    primary_key = ensure_list(dynamic_thresholds.get("PrimaryKey", {}).get("Key", []))
    calculations = ensure_list(dynamic_thresholds.get("Calculation", []))
    
    # Make a copy of the event level dataframe
    result_df = event_level_df.clone()
    
    # Create an empty struct column for derived values if it doesn't exist
    if "derived" not in result_df.columns:
        result_df = result_df.with_columns(pl.lit(None).alias("derived"))
    
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
        dataset_df = datasets[dataset_id]
        
        if lazy:
            dataset_df = dataset_df.lazy()
        
        # Apply filters if any
        filters = ensure_list(calc.get("Filters", {}).get("Filter", []))
        if filters:
            dataset_df = DFoperations.df_filters(dataset_df, filters)
        
        # Process DerivedValues
        derived_values = calc.get("DerivedValues", {})
        column_name = derived_values.get("Column", "")
        
        # Check if GroupFunction exists for group by operation
        group_function = derived_values.get("GroupFunction")
        derived_values_dict = {}
        
        if group_function:
            # Get column and apply group by operation
            dataset_df = DFoperations.df_group_by(
                dataset_df, 
                join_key, 
                column_name,
                [column_name], 
                [group_function]
            )
            
            # The GroupFunction becomes the subindex name
            derived_values_dict[group_function] = dataset_df[column_name]
        else:
            # No group by, just use the column as is
            derived_values_dict[column_name] = dataset_df[column_name]
        
        # Join the dataset to the event_level_df
        result_df = result_df.join(
            dataset_df.select(join_key + list(derived_values_dict.keys())),
            left_on=primary_key,
            right_on=join_key,
            how="left"
        )
        
        # Update derived struct with the new values
        for key, value in derived_values_dict.items():
            result_df = result_df.with_columns(
                pl.struct(["derived"]).update(pl.struct({key: pl.col(key)})).alias("derived")
            )
        
        # Process Values (upper_bound, lower_bound, etc.)
        values = ensure_list(calc.get("Values", {}).get("Value", []))
        for value in values:
            name = value.get("@name")
            val_type = value.get("@type")
            expression = value.get("#text", "")
            
            # Parse and evaluate the expression
            parsed_expr = parse_expression(expression)
            
            # Add the calculated value to the result dataframe
            try:
                result_df = result_df.with_columns(
                    eval(parsed_expr).cast(get_polars_type(val_type)).alias(name)
                )
            except Exception as e:
                print(f"Error evaluating expression '{parsed_expr}': {e}")
        
        # Drop temporary columns used for calculations (except those in Values)
        cols_to_drop = [col for col in derived_values_dict.keys() 
                       if col not in [val.get("@name") for val in values]]
        if cols_to_drop:
            result_df = result_df.drop(cols_to_drop)
    
    return result_df

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
    
    # Load datasets
    datasets = {"ds2": pl.read_csv("C:/Users/h59257/Downloads/profiles_sgp.csv")}
    
    # Create a simple event level dataframe with static thresholds
    event_level_df = pl.DataFrame({
        "account_id": ["1", "2", "3"],
        "static_thresholds": [
            {"std_cycles": 2.5},
            {"std_cycles": 3.0},
            {"std_cycles": 1.8}
        ]
    })
    
    # Track memory usage
    initial_memory = get_memory_usage()
    
    # Run dynamic threshold calculation
    result_df = dynamic_threshold(dynamic_thresholds, datasets, event_level_df, lazy=False)
    
    # Final memory
    final_memory = get_memory_usage()
    
    print(result_df)
    print(f"Memory usage: {final_memory - initial_memory} MB")