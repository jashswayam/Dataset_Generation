for col_name, col_type in column_mapping.items():
            if col_name in DYN_CAL_DF.columns:
                # First rename the column if it exists and needs to be renamed
                # Then cast it to the specified type
                if col_type.lower() == "string":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Utf8).alias(col_name))
                elif col_type.lower() in ["int", "integer", "int64"]:
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Int64).alias(col_name))
                elif col_type.lower() == "int32":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Int32).alias(col_name))
                elif col_type.lower() == "int16":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Int16).alias(col_name))
                elif col_type.lower() == "int8":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Int8).alias(col_name))
                elif col_type.lower() == "uint32":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.UInt32).alias(col_name))
                elif col_type.lower() == "uint64":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.UInt64).alias(col_name))
                elif col_type.lower() in ["float", "double", "float64"]:
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Float64).alias(col_name))
                elif col_type.lower() == "float32":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Float32).alias(col_name))
                elif col_type.lower() == "decimal":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Decimal).alias(col_name))
                elif col_type.lower() == "bool" or col_type.lower() == "boolean":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Boolean).alias(col_name))
                elif col_type.lower() == "date":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Date).alias(col_name))
                elif col_type.lower() == "datetime":
                    DYN_CAL_DF = DYN_CAL_DF.with_column(pl.col(col_name).cast(pl.Datetime).alias(col_name))
    