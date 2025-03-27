import xmltodict
import polars as pl
from Operations import ExtendedOperator  # Assuming this exists for filtering

def Dynamic_Threshold(xml_data: str, datasets: dict, DYN_CAL_DF: pl.DataFrame, lazy: bool = False):
    """
    Parses DynamicThresholdCalculations from XML and updates DYN_CAL_DF.
    
    Parameters:
        xml_data (str): The XML string to parse.
        datasets (dict): A dictionary of DataFrames (dataset_id -> polars.DataFrame).
        DYN_CAL_DF (pl.DataFrame): The main output DataFrame to be modified.
        lazy (bool): Whether to use lazy execution (default: False).
        
    Returns:
        pl.DataFrame: The modified DYN_CAL_DF.
    """
    # Convert XML to dict
    xml_dict = xmltodict.parse(xml_data)
    dynamic_thresholds = xml_dict.get("Rules", {}).get("DynamicThresholdCalculations", {})

    # Extract Primary Key for DYN_CAL_DF
    primary_key = dynamic_thresholds.get("PrimaryKey", {}).get("Key", None)
    if not primary_key or not primary_key.strip():
        raise ValueError("PrimaryKey <Key> cannot be null or empty")

    # Process each Calculation
    calculations = dynamic_thresholds.get("Calculation", [])
    if not isinstance(calculations, list):  # Ensure it's always a list
        calculations = [calculations]

    for calc in calculations:
        dataset_id = calc.get("DatasetId", None)
        join_key = calc.get("Key", None)
        columns = calc.get("Columns", {}).get("Column", [])

        if not dataset_id or not join_key:
            raise ValueError("Each <Calculation> must have a non-empty <DatasetId> and <Key>")

        # Ensure columns is always a list
        if not isinstance(columns, list):
            columns = [columns]

        # Convert column details to a dictionary {name: type}
        column_mapping = {col["@name"]: col["@type"] for col in columns if "@name" in col and "@type" in col}

        # Get the dataset
        if dataset_id not in datasets:
            raise ValueError(f"Dataset {dataset_id} not found in provided DataFrames.")
        dataset_df = datasets[dataset_id]

        # Ensure lazy execution if needed
        if lazy:
            dataset_df = dataset_df.lazy()
            DYN_CAL_DF = DYN_CAL_DF.lazy()

        # Apply filters if present
        filters = calc.get("Filters", {}).get("Filter", [])
        if not isinstance(filters, list):  # Ensure filters is a list
            filters = [filters]

        for flt in filters:
            column, operator, value = flt.get("Column"), flt.get("Operator"), flt.get("Value")
            if column and operator and value:
                operation = getattr(ExtendedOperator, operator, None)
                if operation:
                    dataset_df = dataset_df.filter(operation(pl.col(column), value))
                else:
                    raise ValueError(f"Unsupported operator: {operator}")

        # Select necessary columns
        dataset_df = dataset_df.select([join_key] + list(column_mapping.keys()))

        # Merge with DYN_CAL_DF
        DYN_CAL_DF = DYN_CAL_DF.join(dataset_df, on=join_key, how="left")

    return DYN_CAL_DF.collect() if lazy else DYN_CAL_DF