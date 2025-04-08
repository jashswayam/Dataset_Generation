import polars as pl
from utils import DFoperations, ensure_list

def parse_expression(expression, dataset_columns=None):
    """
    Parse expressions with various prefixes:
    - derived. : refers to derived_values struct
    - dth. : refers to dynamic threshold struct
    - th. : refers to static threshold struct
    - ev. : refers to columns in event level dataframe
    - ds. : refers to columns in the dataset being processed
    
    If no prefix is present, treat as a literal value.
    """
    # First attempt to evaluate directly to handle literals
    try:
        # Check if it's a numeric literal or boolean
        if expression.lower() in ('true', 'false'):
            return expression.lower() == 'true'
        
        value = eval(expression)
        if isinstance(value, (int, float, bool, str, tuple, list)):
            return value
    except:
        pass
    
    # Handle prefixed expressions
    if '.' in expression:
        prefix, field = expression.split('.', 1)
        
        if prefix == 'derived':
            return f"pl.col('derived_values').struct.field('{field}')"
        elif prefix == 'dth':
            return f"pl.col('dynamic_thresholds').struct.field('{field}')"
        elif prefix == 'th':
            return f"pl.col('static_thresholds').struct.field('{field}')"
        elif prefix == 'ev':
            return f"pl.col('{field}')"
        elif prefix == 'ds' and dataset_columns and field in dataset_columns:
            return f"pl.col('{field}')"
    
    # If no special prefix and not a literal, return as is
    return expression

def evaluate_expression(df, expression, dataset_columns=None):
    """
    Evaluate a parsed expression against a dataframe
    """
    parsed_expr = parse_expression(expression, dataset_columns)
    
    # If the parsed expression is a literal value, return it
    if not isinstance(parsed_expr, str) or not parsed_expr.startswith("pl."):
        return parsed_expr
    
    # Otherwise evaluate the Polars expression
    try:
        return eval(parsed_expr)
    except Exception as e:
        print(f"Error evaluating expression '{parsed_expr}': {e}")
        return None

def dynamic_threshold(dynamic_thresholds: dict, datasets: dict, event_level_dict: dict, lazy: bool = False):
    """
    Process dynamic thresholds calculations according to the XML structure.
    
    Parameters:
    - dynamic_thresholds: The parsed XML containing the DynamicThresholdCalculations
    - datasets: Dictionary of dataframes referenced in the calculation
    - event_level_dict: Dictionary with event level dataframes and their primary keys
    - lazy: Whether to use lazy evaluation
    
    Returns:
    - Updated event level dataframes dictionary
    """
    calculations = ensure_list(dynamic_thresholds.get("Calculation", []))
    result_dict = {}
    
    # Process each event level dataframe
    for event_key, event_data in event_level_dict.items():
        event_df = event_data["dataframe"]
        primary_key = event_data["primary_key"]
        
        # Make a copy of the event level dataframe
        result_df = event_df.clone()
        
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
            dataset_columns = dataset_df.columns
            
            if lazy:
                dataset_df = dataset_df.lazy()
            
            # Apply filters if any
            filters = ensure_list(calc.get("Filters", {}).get("Filter", []))
            if filters:
                filtered_dataset_df = dataset_df
                for filter_def in filters:
                    column = filter_def.get("Column", "")
                    operator = filter_def.get("Operator", "")
                    value_text = filter_def.get("Value", "")
                    
                    # Parse value using the expression parser
                    value = evaluate_expression(filtered_dataset_df, value_text, dataset_columns)
                    
                    # Apply filter based on operator
                    if operator.lower() == "eq":
                        filtered_dataset_df = filtered_dataset_df.filter(pl.col(column) == value)
                    elif operator.lower() == "between":
                        if isinstance(value, tuple) and len(value) == 2:
                            filtered_dataset_df = filtered_dataset_df.filter(
                                (pl.col(column) >= value[0]) & (pl.col(column) <= value[1])
                            )
                        else:
                            print(f"Warning: 'between' operator requires a tuple of two values")
                    # Add more operators as needed
                
                dataset_df = filtered_dataset_df
            
            # Process DerivedValues
            derived_values = calc.get("DerivedValues", {})
            column_name = derived_values.get("Column", "")
            
            # Check if GroupFunction exists for group by operation
            group_function = derived_values.get("GroupFunction")
            derived_values_dict = {}
            
            if group_function:
                # Handle multiple group functions if comma-separated
                group_functions = [func.strip() for func in group_function.split(",")]
                
                # Apply group by for each function
                grouped_df = dataset_df
                for func in group_functions:
                    grouped_df = DFoperations.df_group_by(
                        grouped_df, 
                        join_key, 
                        column_name,
                        [column_name], 
                        [func]
                    )
                    
                    # Store each result in derived_values_dict
                    result_column = f"{column_name}_{func}" if func != "count" else "count"
                    derived_values_dict[func] = grouped_df[result_column]
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
            
            # Update derived_values struct with the new values
            for key, value in derived_values_dict.items():
                result_df = result_df.with_columns(
                    pl.struct(["derived_values"]).update(pl.struct({key: pl.col(key)})).alias("derived_values")
                )
            
            # Process Values (upper_bound, lower_bound, etc.) and store in dynamic_thresholds
            values = ensure_list(calc.get("Values", {}).get("Value", []))
            dth_updates = {}
            
            for value in values:
                name = value.get("@name")
                val_type = value.get("@type")
                expression = value.get("#text", "")
                
                # Evaluate the expression
                try:
                    result_value = evaluate_expression(result_df, expression, dataset_columns)
                    dth_updates[name] = result_value.cast(get_polars_type(val_type))
                except Exception as e:
                    print(f"Error evaluating expression '{expression}': {e}")
            
            # Update dynamic_thresholds struct with calculated values
            if dth_updates:
                result_df = result_df.with_columns(
                    pl.struct(["dynamic_thresholds"]).update(pl.struct(dth_updates)).alias("dynamic_thresholds")
                )
            
            # Drop temporary columns used for calculations
            cols_to_drop = list(derived_values_dict.keys())
            if cols_to_drop:
                result_df = result_df.drop(cols_to_drop)
        
        # Store the updated dataframe in the result dictionary
        result_dict[event_key] = {
            "dataframe": result_df,
            "primary_key": primary_key
        }
    
    return result_dict

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
    with open('rule_2.xml', 'r', encoding="utf-8") as file:
        xml_data = file.read()
    
    # Parse XML
    xml_dict = xmltodict.parse(xml_data)
    dynamic_thresholds = xml_dict.get("Rule").get("DynamicThresholdCalculations")
    
    # Load datasets
    datasets = {
        "ds2": pl.read_csv("profiles_sgp.csv")
    }
    
    # Load event level dataframes from parquet files
    event_level_dict = {
        "event1": {
            "dataframe": pl.read_parquet("event_level1.parquet"),
            "primary_key": ["account_id"]
        },
        "event2": {
            "dataframe": pl.read_parquet("event_level2.parquet"),
            "primary_key": ["account_id"]
        }
    }
    
    # Track memory usage
    initial_memory = get_memory_usage()
    
    # Run dynamic threshold calculation
    result_dict = dynamic_threshold(dynamic_thresholds, datasets, event_level_dict, lazy=False)
    
    # Final memory
    final_memory = get_memory_usage()
    
    # Print results
    for event_key, event_data in result_dict.items():
        print(f"\nEvent: {event_key}")
        print(event_data["dataframe"])
    
    print(f"\nMemory usage: {final_memory - initial_memory} MB")