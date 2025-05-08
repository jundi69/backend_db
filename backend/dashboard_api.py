import os

from dotenv import load_dotenv

load_dotenv()
import json
import math
from datetime import datetime

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient
from influxdb_client.client.flux_table import FluxStructureEncoder

app = FastAPI(title="Distributed Training Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize InfluxDB client
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "your-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "distributed-training")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "distributed-training-metrics")
print(f"Using InfluxDB URL: {INFLUXDB_URL}")
print(f"Using InfluxDB Token: {INFLUXDB_TOKEN}")
print(f"Using InfluxDB Org: {INFLUXDB_ORG}")
print(f"Using InfluxDB Bucket: {INFLUXDB_BUCKET}")

influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = influx_client.query_api()

# Initialize Redis for caching
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL)
CACHE_TTL = 60  # seconds


@app.get("/")
async def root():
    return {"message": "Distributed Training Dashboard API"}


def process_multi_miner_field_timeseries(tables, field_name_to_extract="loss"):
    """
    Processes Flux query results for a specific field across multiple miners.
    Returns a dictionary where keys are miner_uids and values are their time-series data for the field.
    Example: { "miner_122": [{"time": "...", "value": 0.5}, ...], "miner_123": [...] }
    """
    miner_data_map = {}

    # The Flux query should ideally group data in a way that makes processing easier.
    # Let's assume the query gives us tables where each table might contain records for multiple miners,
    # or one table per miner if grouped by miner_uid first.
    # The crucial part is that each record has 'miner_uid', '_time', and '_value' for the target field.

    # Assuming `tables` is the direct result from a query like:
    # from(bucket: "...")
    #   |> range(...)
    #   |> filter(fn: (r) => r._measurement == "training_metrics" and r._field == "loss")
    #   |> keep(columns: ["_time", "_value", "miner_uid"]) // Ensure miner_uid is present

    for table in tables: # `tables` is the raw JSON from FluxStructureEncoder if you used it.
                         # If it's the direct InfluxDB client result, it's a list of FluxTable.
        for record_data in table.get("records", []): # If using FluxStructureEncoder output
        # for record in table.records: # If using direct InfluxDB client FluxTable list
            # miner_uid = record.values.get("miner_uid")
            # dt_obj = record.values.get("_time")
            # value = record.values.get("_value")
            miner_uid = record_data.get("miner_uid") # From FluxStructureEncoder output
            dt_obj = record_data.get("_time")       # From FluxStructureEncoder output
            value = record_data.get("_value")         # From FluxStructureEncoder output

            if miner_uid is None or dt_obj is None or value is None:
                continue # Skip records with missing essential data

            # Convert time if it's a datetime object (it should be if coming from InfluxDB client directly)
            # If using FluxStructureEncoder, it might already be a string.
            if isinstance(dt_obj, datetime):
                time_str = dt_obj.isoformat()
            else: # Assume it's already a string (e.g. from FluxStructureEncoder)
                time_str = str(dt_obj)


            if miner_uid not in miner_data_map:
                miner_data_map[miner_uid] = []
            
            miner_data_map[miner_uid].append({"time": time_str, "value": float(value)}) # Ensure value is float

    # Sort each miner's time-series
    for uid in miner_data_map:
        miner_data_map[uid] = sorted(miner_data_map[uid], key=lambda x: x["time"])
    
    return miner_data_map

@app.get("/metrics/global")
async def get_global_metrics(time_range: str = "1h"):
    """Get global training metrics for all miners"""
    cache_key = f"global_metrics:{time_range}"
    cached = redis_client.get(cache_key)

    if cached:
        return json.loads(cached)

    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r._measurement == "global_training_metrics")
        |> group(columns: ["_time", "_field"])
        |> yield(name: "global_metrics")
    '''

    try:
        result = query_api.query(query)
        tables = json.loads(json.dumps(result, cls=FluxStructureEncoder))

        # Process the data for frontend consumption
        processed_data = process_metrics_for_dashboard(tables)

        # Cache the result
        redis_client.setex(cache_key, CACHE_TTL, json.dumps(processed_data))
        return processed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/miners")
async def get_miners_list():
    """Get a list of all miner UIDs"""
    cache_key = "miners_list"
    cached = redis_client.get(cache_key)

    if cached:
        return json.loads(cached)

    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -1h)
        |> filter(fn: (r) => r._measurement == "metagraph_metrics")
        |> group(columns: ["miner_uid"])
        |> distinct(column: "miner_uid")
    '''

    try:
        result = query_api.query(query)
        miners = []

        for table in result:
            for record in table.records:
                miners.append(record.values.get("miner_uid"))

        # Cache the result
        redis_client.setex(cache_key, CACHE_TTL, json.dumps(miners))
        return miners
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/miner/{uid}")
async def get_miner_metrics(
    uid: int,
    time_range: str = "1h",
    include_scores: bool = True,
    include_training: bool = True,
    include_resources: bool = True,
):
    """Get metrics for a specific miner"""
    cache_key = f"miner:{uid}:{time_range}:{include_scores}:{include_training}:{include_resources}"
    cached = redis_client.get(cache_key)

    if cached:
        return json.loads(cached)

    result = {}

    try:
        # Get metagraph metrics
        metagraph_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: -{time_range})
            |> filter(fn: (r) => r._measurement == "metagraph_metrics" and r.miner_uid == "{uid}")
            |> last()
        '''
        metagraph_result = query_api.query(metagraph_query)
        result["metagraph"] = process_metagraph_metrics(metagraph_result)

        # Get incentive metrics
        incentive_series_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: -{time_range})
            |> filter(fn: (r) => r._measurement == "metagraph_metrics" and r.miner_uid == "{uid}" and r._field == "incentive")
            |> keep(columns: ["_time", "_value"]) // Keep only necessary columns
        '''
        incentive_series_result = query_api.query(incentive_series_query)

        processed_incentive_series = []
        for table in incentive_series_result:  # Should be one table for "incentive"
            for record in table.records:
                dt_obj = record.values.get("_time")
                time_str = (
                    dt_obj.isoformat() if isinstance(dt_obj, datetime) else str(dt_obj)
                )
                processed_incentive_series.append(
                    {"time": time_str, "value": record.values.get("_value")}
                )
        result["incentive_timeseries"] = sorted(
            processed_incentive_series, key=lambda x: x["time"]
        )

        # Get scoring metrics if requested
        if include_scores:
            scores_query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: -{time_range})
                |> filter(fn: (r) => r._measurement == "miner_scores" and r.miner_uid == "{uid}")
                |> group(columns: ["_field", "validator_uid"])
            '''
            scores_result = query_api.query(scores_query)
            result["scores"] = process_scoring_metrics(scores_result)

        # Get training metrics if requested
        if include_training:
            training_query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: -{time_range})
                |> filter(fn: (r) => r._measurement == "training_metrics" and r.miner_uid == "{uid}")
                |> group(columns: ["_field"])
            '''
            training_result = query_api.query(training_query)
            result["training"] = process_training_metrics(training_result)

        # Get resource metrics if requested
        if include_resources:
            resource_query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: -{time_range})
                |> filter(fn: (r) => r._measurement == "resource_metrics" and r.miner_uid == "{uid}")
                |> group(columns: ["_field"])
            '''
            resource_result = query_api.query(resource_query)
            result["resources"] = process_resource_metrics(resource_result)

        # Cache the result
        redis_client.setex(cache_key, CACHE_TTL, json.dumps(result))
        return result

    except Exception as e:
        import traceback

        print("----------- ERROR IN get_miner_metrics -----------")
        traceback.print_exc()  # This will print the full traceback to your FastAPI console
        print(f"Exception type: {type(e)}")
        print(f"Exception details: {str(e)}")
        print("----------------------------------------------------")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/validators")
async def get_validators_list():
    """Get a list of all validator UIDs"""
    cache_key = "validators_list"
    cached = redis_client.get(cache_key)

    if cached:
        return json.loads(cached)

    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -1h)
        |> filter(fn: (r) => r._measurement == "miner_scores")
        |> group(columns: ["validator_uid"])
        |> distinct(column: "validator_uid")
    '''

    try:
        result = query_api.query(query)
        validators = []

        for table in result:
            for record in table.records:
                validators.append(record.values.get("validator_uid"))

        # Cache the result
        redis_client.setex(cache_key, CACHE_TTL, json.dumps(validators))
        return validators
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/validator/{uid}")
async def get_validator_metrics(uid: int, time_range: str = "1h"):
    """Get scoring metrics from a specific validator"""
    cache_key = f"validator:{uid}:{time_range}"
    cached = redis_client.get(cache_key)

    if cached:
        return json.loads(cached)

    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r._measurement == "miner_scores" and r.validator_uid == "{uid}")
        |> group(columns: ["miner_uid", "_field"])
        |> last()
    '''

    try:
        result = query_api.query(query)
        processed_data = process_validator_metrics(result)

        # Cache the result
        redis_client.setex(cache_key, CACHE_TTL, json.dumps(processed_data))
        return processed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/allreduce")
async def get_allreduce_metrics(time_range: str = "1d"):
    """Get metrics about AllReduce operations"""
    cache_key = f"allreduce:{time_range}"
    cached = redis_client.get(cache_key)

    if cached:
        return json.loads(cached)

    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r._measurement == "allreduce_operations")
        |> group(columns: ["epoch", "operation_id"])
    '''

    try:
        result = query_api.query(query)
        processed_data = process_allreduce_metrics(result)

        # Cache the result
        redis_client.setex(cache_key, CACHE_TTL, json.dumps(processed_data))
        return processed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions to process InfluxDB data for frontend consumption
def process_metrics_for_dashboard(tables):
    """Process global metrics for dashboard display"""
    loss_data = extract_time_series(tables, "loss")

    # Calculate perplexity from loss
    perplexity_data = []
    for item in loss_data:
        perplexity_data.append({"time": item["time"], "value": math.exp(item["value"])})

    return {
        "epochs": extract_time_series(tables, "epoch"),
        "loss": loss_data,
        "perplexity": perplexity_data,
        "training_rate": extract_time_series(tables, "training_rate"),
        "bandwidth": extract_time_series(tables, "bandwidth"),
        "active_miners": extract_time_series(tables, "active_miners"),
    }


def extract_time_series(tables, field_name):
    """Extract time series data for a specific field"""
    data = []

    for table in tables:
        if table.get("name") == field_name:
            for record in table.get("records", []):
                data.append(
                    {"time": record.get("_time"), "value": record.get("_value")}
                )

    return sorted(data, key=lambda x: x["time"])


def process_metagraph_metrics(result):
    """Process metagraph metrics for a miner"""
    metrics = {}

    for table in result:
        for record in table.records:
            field = record.values.get("_field")
            value = record.values.get("_value")
            metrics[field] = value

    return metrics


def process_scoring_metrics(result):
    """Process scoring metrics from validators"""
    scores = {}

    for table in result:
        for record in table.records:
            validator = record.values.get("validator_uid")
            field = record.values.get("_field")
            value = record.values.get("_value")

            if validator not in scores:
                scores[validator] = {}

            scores[validator][field] = value

    return scores


def process_single_field_timeseries(result):
    """
    Processes a query result for a single field expected to be a time series.
    Assumes the query was already filtered for the specific field
    or that the result contains tables, one of which is the desired field.
    """
    data_points = []
    for table in result:
        for record in table.records:
            dt_obj = record.values.get("_time")
            time_str = (
                dt_obj.isoformat() if isinstance(dt_obj, datetime) else str(dt_obj)
            )
            data_points.append({"time": time_str, "value": record.values.get("_value")})
    return sorted(data_points, key=lambda x: x["time"])


def process_training_metrics(result):
    """Process training metrics for a miner"""
    training_data = {}
    for table in result:
        field_name = None
        time_series = []
        for record in table.records:
            if field_name is None:
                field_name = record.values.get("_field")

            # Get the datetime object
            dt_obj = record.values.get("_time")
            # Convert to ISO 8601 string if it's a datetime object
            time_str = (
                dt_obj.isoformat() if isinstance(dt_obj, datetime) else str(dt_obj)
            )  # Handle if it's already a string or None

            time_series.append(
                {
                    "time": time_str,  # Store as string
                    "value": record.values.get("_value"),
                }
            )
        if field_name:
            training_data[field_name] = sorted(
                time_series, key=lambda x: x["time"]
            )  # Sorting strings might be okay if ISO format
    return training_data


def process_resource_metrics(result):
    """Process resource metrics for a miner"""
    resource_data = {}
    for table in result:
        field_name = None
        time_series = []
        for record in table.records:
            if field_name is None:
                field_name = record.values.get("_field")

            dt_obj = record.values.get("_time")
            time_str = (
                dt_obj.isoformat() if isinstance(dt_obj, datetime) else str(dt_obj)
            )

            time_series.append(
                {
                    "time": time_str,  # Store as string
                    "value": record.values.get("_value"),
                }
            )
        if field_name:
            resource_data[field_name] = sorted(time_series, key=lambda x: x["time"])
    return resource_data


def process_validator_metrics(result):
    """Process validator scoring data"""
    miner_scores = {}

    for table in result:
        for record in table.records:
            miner = record.values.get("miner_uid")
            field = record.values.get("_field")
            value = record.values.get("_value")

            if miner not in miner_scores:
                miner_scores[miner] = {}

            miner_scores[miner][field] = value

    return miner_scores


def process_allreduce_metrics(result):
    """Process AllReduce operation metrics"""
    operations = []

    # Group by operation_id
    op_groups = {}

    for table in result:
        for record in table.records:
            op_id = record.values.get("operation_id")
            field = record.values.get("_field")
            value = record.values.get("_value")
            time = record.values.get("_time")
            epoch = record.values.get("epoch")

            if op_id not in op_groups:
                op_groups[op_id] = {
                    "operation_id": op_id,
                    "epoch": epoch,
                    "time": time,
                    "metrics": {},
                }

            op_groups[op_id]["metrics"][field] = value

    # Convert to list
    for op_id, data in op_groups.items():
        operations.append(data)

    # Sort by time
    return sorted(operations, key=lambda x: x["time"], reverse=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
