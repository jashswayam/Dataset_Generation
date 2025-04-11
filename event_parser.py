import polars as pl
import re
import xmltodict
import warnings

warnings.filterwarnings("ignore")

def event_level_parser(pxml_event_level: dict, datasets_dict: dict) -> dict:
    columns = pxml_event_level.get("RetainedColumns").get("Column")
    return event_level(pxml_event_level, datasets_dict)

def extract_dataset(event_columns: list[str], datasets_dict: dict):
    dataset_map = {}
    for item in event_columns:
        ds, col = item.split('.', maxsplit=1)
        keys = datasets_dict.get(ds).get("joining_keys")
        dataset_map.setdefault(ds, keys)
    return dataset_map

def extract_ds_columns(dataset_map: str):
    # Only match ds-prefixed dataset references like ds1.colX
    pattern = r'\b(ds\d+)\.(\w+)\b'
    matches = re.findall(pattern, dataset_map)
    selective_dataset = {}
    for ds, col in matches:
        selective_dataset.setdefault(ds, set()).add(col)
    return selective_dataset

def build_expression(dataset_map):
    datasets = list(dataset_map.keys())
    expr = f"{datasets[0]}"
    for ds in datasets[0:]:
        print(ds)
        on = dataset_map.get(ds)
        expr += f".join({ds}, left_on={dataset_map.get(ds)}, " \
                f"right_on={dataset_map.get(ds + 1)}, " \
                f"coalesce=True, suffix = {'_' + ds} ')"
    return expr

def event_level(pxml_event_level, datasets_dict):
    # Extract required information from XML
    columns = pxml_event_level.get("RetainedColumns").get("Column")
    
    # Extract dataset information
    dataset_map = extract_dataset(columns, datasets_dict)
    
    # Build expression for joining datasets
    expression = build_expression(dataset_map)
    
    # Execute the expression (you might need to implement this part)
    # This would involve actually performing the join operations
    
    # For demonstration, just return the expression
    return {
        "expression": expression,
        "datasets": dataset_map
    }

# If running as main script
if __name__ == "__main__":
    # Load XML file
    with open("C:/Users/h59257/Downloads/rule_2.xml", 'r', encoding="utf-8") as file:
        xml_data = file.read()
    
    # Define datasets dictionary
    datasets_dict = {
        "ds1": {
            "joining_keys": "[account_key]",
            "dataframe": pl.read_csv("C:/Users/h59257/Downloads/ds1_sgp.csv")
        },
        "ds2": {
            "joining_keys": "[account_id]",
            "dataframe": pl.read_csv("C:/Users/h59257/Downloads/profiles_sgp.csv")
        },
        "ds3": {
            "joining_keys": "[account_id]",
            "dataframe": pl.read_csv("C:/Users/h59257/Downloads/profiles_sgp.csv")
        }
    }
    
    # Parse XML
    xml_dict = xmltodict.parse(xml_data)
    pxml_event_level = xml_dict.get("Rule").get("EventLevel")
    
    # Get result
    result = event_level(pxml_event_level, datasets_dict)
    print(result)