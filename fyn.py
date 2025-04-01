import xmltodict
import polars as pl
import ast
from operations import ExtendedOperator  # Assuming this exists for filtering


class DataFilter:
    """Class to handle filtering operations on DataFrames."""
    
    @staticmethod
    def apply_filters(dataset_df, filters):
        """
        Apply filters to a dataframe.
        
        Args:
            dataset_df: DataFrame to filter
            filters: List of filter conditions from XML
            
        Returns:
            Filtered DataFrame
        """
        if not filters:
            return dataset_df
            
        if not isinstance(filters, list):  # Ensure filters is a list
            filters = [filters]
            
        filtered_df = dataset_df
        for flt in filters:
            column, operator, value = flt.get("Column"), flt.get("Operator"), flt.get("Value")
            if column and operator and value:
                operation = getattr(ExtendedOperator, operator, None)
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    value = value
                    
                if operation:
                    filtered_df = filtered_df.filter(operation(filtered_df.select(column).collect()[column], value))
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
                    
        return filtered_df


class DataGrouper:
    """Class to handle grouping operations on DataFrames."""
    
    @staticmethod
    def apply_group_by(dataset_df, group_col, functions, join_key):
        """
        Apply group by operations to a dataframe.
        
        Args:
            dataset_df: DataFrame to group
            group_col: Column to group by
            functions: Comma-separated string of functions to apply
            join_key: Key used for joining
            
        Returns:
            Grouped DataFrame
        """
        # Convert function string to list
        function_list = [func.strip() for func in functions.split(",")]
        
        # Apply aggregation functions directly - let Polars handle errors
        agg_exprs = []
        for func in function_list:
            # Dynamically call the function on the column
            # This will throw a Polars error if the function doesn't exist
            agg_exprs.append(getattr(pl.col(group_col), func)().alias(f"{group_col}_{func}"))
        
        grouped_df = dataset_df.group_by(join_key).agg(agg_exprs)
        return grouped_df.select(list(grouped_df.columns))


class ColumnRenamer:
    """Class to handle column renaming operations."""
    
    @staticmethod
    def rename_columns(df, column_mapping, join_key_mapping=None):
        """
        Rename columns according to the XML configuration.
        
        Args:
            df: DataFrame to rename columns
            column_mapping: Dictionary mapping original column names to desired names
            join_key_mapping: Mapping for join key renaming
            
        Returns:
            DataFrame with renamed columns
        """
        # Implementation will depend on your specific requirements for renaming
        rename_dict = {}
        
        # Process column mappings
        for original_name, col_info in column_mapping.items():
            if "@name" in col_info:
                new_name = col_info["@name"]
                rename_dict[original_name] = new_name
        
        # Process join key mapping if provided
        if join_key_mapping and join_key_mapping in df.columns:
            original_key = join_key_mapping
            if original_key in df.columns:
                rename_dict[original_key] = column_mapping.get(original_key, {}).get("@name", original_key)
        
        # Apply renaming
        if rename_dict:
            df = df.rename(rename_dict)
            
        return df


def dynamic_threshold(xml_data: str, datasets: dict, lazy: bool = False):
    """
    Calculate dynamic thresholds based on XML configuration.
    
    Args:
        xml_data: XML string containing calculation rules
        datasets: Dictionary of datasets {dataset_id: polars_dataframe}
        lazy: Whether to use lazy execution
        
    Returns:
        DataFrame with calculated values
    """
    # Convert XML to dict
    xml_dict = xmltodict.parse(xml_data)
    
    dynamic_thresholds = xml_dict.get("Rule", {}).get("DynamicThresholdCalculations", {})
    
    # Extract Primary Key for DYN_CAL_DF
    primary_key = dynamic_thresholds.get("PrimaryKey", {}).get("Key", None)
    
    if not primary_key or not primary_key.strip():
        raise ValueError("PrimaryKey <Key> cannot be null or empty")
    
    # Process each Calculation
    calculations = dynamic_thresholds.get("Calculation", [])
    
    if not isinstance(calculations, list):  # Ensure it's always a list
        calculations = [calculations]
    
    dyn_cal_df = pl.DataFrame()
    initial_flag = True
    column_mapping = {}  # Store column mappings for final renaming
    join_key_mapping = None
    
    for calc in calculations:
        dataset_id = calc.get("DatasetId")
        join_key = calc.get("Keys", {}).get("Key")
        
        # Store join key for later renaming
        if join_key and not join_key_mapping:
            join_key_mapping = join_key
            
        columns = calc.get("Columns", {}).get("Column", [])
        
        # Ensure columns is always a list
        if not isinstance(columns, list):
            columns = [columns]
            
        # Convert column details to a dictionary {name: column_info}
        for col in columns:
            if "@name" in col:
                column_mapping[col["@name"]] = col
        
        dataset_df = datasets[dataset_id]
        
        # Ensure lazy execution if needed
        if lazy:
            dataset_df = dataset_df.lazy()
            dyn_cal_df = dyn_cal_df.lazy() if not dyn_cal_df.is_empty() else dyn_cal_df
        
        # Apply filters if present
        filters = calc.get("Filters", {}).get("Filter", [])
        dataset_df = DataFilter.apply_filters(dataset_df, filters)
        
        # Apply Group By if present
        value_section = calc.get("Value")
        group_by_section = value_section.get("GroupBy") if value_section else None
        
        if group_by_section:
            group_col = group_by_section.get("Column")
            functions = group_by_section.get("Function")
            dataset_df = DataGrouper.apply_group_by(dataset_df, group_col, functions, join_key)
        
        # Merge with dyn_cal_df
        if initial_flag:
            dyn_cal_df = dataset_df
            initial_flag = False
        else:
            dyn_cal_df = dyn_cal_df.join(dataset_df, on=join_key, how="left")
    
    # Apply final column renaming
    dyn_cal_df = ColumnRenamer.rename_columns(dyn_cal_df, column_mapping, join_key_mapping)
    
    return dyn_cal_df


if __name__ == "__main__":
    with open('C:/Users/h59257/Downloads/rule_2.xml', 'r', encoding='utf-8') as file:
        xml_data = file.read()
        
    datasets = {"ds2": pl.read_csv("C:/Users/h59257/Downloads/profiles_sgp.csv")}
    
    dyn_cal_df = dynamic_threshold(xml_data, datasets, lazy=True).collect()
    print(dyn_cal_df)