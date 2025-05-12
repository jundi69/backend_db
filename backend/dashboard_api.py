import os

from dotenv import load_dotenv

load_dotenv()
import asyncio
import json
import math
import traceback
from collections import defaultdict
from datetime import datetime

import geoip2.database
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

try:
    import bittensor as bt

    metagraph = bt.metagraph(netuid=38)  # Replace with your actual netuid if needed
    print("Successfully initialized bittensor metagraph.")
except ImportError:
    print(
        "Warning: bittensor library not found or metagraph sync failed. Location updates might not work."
    )
    metagraph = None
except Exception as e:
    print(f"Warning: Error initializing bittensor metagraph: {e}")
    metagraph = None


# --- GeoIP Configuration ---
GEOLITE2_DB_PATH = os.getenv(
    "GEOLITE2_DB_PATH", "data/GeoLite2-City.mmdb"
)  # Default path
LOCATION_UPDATE_INTERVAL_SECONDS = 300  # Update locations every 5 minutes

try:
    geoip_reader = geoip2.database.Reader(GEOLITE2_DB_PATH)
    print(f"Successfully loaded GeoIP database from: {GEOLITE2_DB_PATH}")
except FileNotFoundError:
    print(
        f"ERROR: GeoLite2 database not found at {GEOLITE2_DB_PATH}. Miner location features will be disabled."
    )
    geoip_reader = None
except Exception as e:
    print(f"ERROR: Failed to load GeoLite2 database: {e}")
    geoip_reader = None


async def update_all_miner_locations():
    """Periodically fetches miner IPs and updates their geo-location in Redis."""
    if not geoip_reader:
        print("GeoIP Reader not available, skipping location update.")
        return
    if not metagraph:
        print("Metagraph not available, skipping location update.")
        return
    if metagraph.n == 0: # Check if metagraph has any neurons
        print("Metagraph is empty, skipping location update.")
        return

    print("Starting periodic miner location update...")
    try:
        # Optional: Ensure metagraph is synced if necessary
        # Consider adding metagraph.sync() here if needed, handling potential errors
        # try:
        #    metagraph.sync()
        #    print("Metagraph synced successfully.")
        # except Exception as e:
        #    print(f"Warning: Failed to sync metagraph during location update: {e}")
        #    # Decide if you want to proceed with potentially stale data or return

        locations_updated = 0
        # Iterate through all UIDs from 0 to n-1
        for uid in range(metagraph.n):
            try:
                # Get the AxonInfo using the UID as the index
                axon = metagraph.axons[uid]

                # Check if this specific UID is serving
                if not axon.is_serving:
                    # Optional: log skipping non-serving UIDs
                    # print(f"Skipping UID {uid}: Not serving.")
                    continue

                # --- Get IP Address (from the AxonInfo object) ---
                ip_address = axon.ip
                # ---------------------------------------------------------

                if not ip_address or ip_address == "0.0.0.0" or ip_address.startswith("192.168.") or ip_address.startswith("10.") or ":" in ip_address: # Also skip IPv6 for now
                    # Optional: log skipping invalid/local IPs
                    # print(f"Skipping location update for UID {uid}: Invalid, private, or IPv6 IP {ip_address}")
                    continue # Skip invalid, local, or IPv6 IPs for simplicity with GeoLite2 City

                # Perform GeoIP lookup
                try:
                    response = geoip_reader.city(ip_address)
                    lat = response.location.latitude
                    lon = response.location.longitude
                    city = response.city.name or "Unknown City"
                    country = response.country.name or "Unknown Country"

                    if lat is not None and lon is not None:
                        # Store in Redis Hash (overwrite previous)
                        redis_key = f"miner_location:{uid}"
                        redis_client.hmset(redis_key, {
                            "uid": uid,
                            "lat": lat,
                            "lon": lon,
                            "city": city,
                            "country": country,
                            "ip": ip_address, # Store IP for debugging?
                            "last_updated": datetime.utcnow().isoformat()
                        })
                        # Optional: log successful update
                        print(f"Updated location for UID {uid} ({ip_address}): {city}, {country} ({lat:.4f}, {lon:.4f})")
                        locations_updated += 1
                    else:
                         # Optional: log missing lat/lon in response
                         print(f"Skipping location update for UID {uid} ({ip_address}): Lat/Lon not found in GeoIP response.")
                         pass # Continue to next UID


                except geoip2.errors.AddressNotFoundError:
                    # Optional: log IP not found in DB
                    print(f"IP address not found in GeoIP database: {ip_address} (UID: {uid})")
                    # Optionally clear old location data?
                    # redis_client.delete(f"miner_location:{uid}")
                    pass # Continue to next UID
                except Exception as e:
                    print(f"Error during GeoIP lookup for {ip_address} (UID: {uid}): {e}")

            except IndexError:
                 # This might happen if metagraph.n changes during iteration, though unlikely with range(metagraph.n)
                 print(f"IndexError accessing axon for UID {uid}. Metagraph size might have changed unexpectedly.")
            except Exception as e:
                print(f"Error processing UID {uid} for location update: {e}")
                # traceback.print_exc() # Uncomment for more detailed debugging if needed

        print(f"Finished miner location update. Updated {locations_updated} locations.")

    except Exception as e:
        print(f"ERROR during periodic location update task: {e}")
        traceback.print_exc()


async def periodic_location_updater():
    """Runs the update task periodically."""
    while True:
        await update_all_miner_locations()
        await asyncio.sleep(LOCATION_UPDATE_INTERVAL_SECONDS)


@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    # Start the periodic location updater in the background
    # Note: For production, consider a more robust scheduler like APScheduler or Celery
    asyncio.create_task(periodic_location_updater())
    print("Started periodic location update task.")


@app.get("/")
async def root():
    return {"message": "Distributed Training Dashboard API"}


# dashboard_api.py


def get_all_miner_timeseries_for_field(
    measurement: str,
    field_name: str,
    time_range: str,
    query_api_instance,
    bucket_name: str,
):
    """
    Fetches and processes time-series data for a specific field from a given measurement
    across all miners.
    Returns a dictionary: { "miner_uid1": [{"time": t, "value": v}, ...], "miner_uid2": [...] }
    """
    # print(f"DEBUG: Fetching for measurement='{measurement}', field='{field_name}', range='{time_range}'")
    query = f'''
    from(bucket: "{bucket_name}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "{field_name}")
        |> keep(columns: ["_time", "_value", "miner_uid"])
        |> group()
    '''
    # print(f"DEBUG: Flux Query:\n{query}")
    raw_result = query_api_instance.query(query)
    try:
        tables_json = json.loads(json.dumps(raw_result, cls=FluxStructureEncoder))
        # print(f"DEBUG: tables_json from FluxStructureEncoder: {json.dumps(tables_json, indent=2)}")
    except Exception as e:
        print(f"DEBUG: Error during FluxStructureEncoder processing: {e}")
        tables_json = []

    miner_data_map = defaultdict(list)

    # if not tables_json:
    # print(f"DEBUG: tables_json is empty. No data from InfluxDB query for {field_name}?")

    for table_data in tables_json:
        # print(f"DEBUG: Processing table_data with keys: {list(table_data.keys())}")
        records = table_data.get("records", [])
        # if not records:
        # print(f"DEBUG: No records in this table_data.")
        for record_data in records:
            # print(f"DEBUG: Full record_data: {record_data}")

            # Corrected extraction: All relevant fields are inside the 'values' sub-dictionary
            record_values = record_data.get("values", {})

            miner_uid = record_values.get("miner_uid")
            dt_obj = record_values.get("_time")  # _time from values
            value = record_values.get("_value")  # _value from values

            # print(f"DEBUG: Extracted - miner_uid: {miner_uid}, time: {dt_obj}, value: {value}")

            if miner_uid is None or dt_obj is None or value is None:
                # print(f"DEBUG: Skipping record due to missing data from 'values' dict: {record_values}")
                continue

            time_str = str(
                dt_obj
            )  # FluxStructureEncoder usually gives ISO strings for time
            try:
                processed_value = float(value) if field_name != "epoch" else int(value)
            except ValueError:
                # print(f"DEBUG: ValueError converting value '{value}' for field '{field_name}', miner '{miner_uid}'. Skipping.")
                continue

            miner_data_map[miner_uid].append(
                {"time": time_str, "value": processed_value}
            )

    # if not miner_data_map:
    # print(f"DEBUG: miner_data_map is empty after processing all records for {field_name}.")
    # else:
    # print(f"DEBUG: miner_data_map for {field_name} before sorting: {dict(miner_data_map)}")

    for uid_key in list(miner_data_map.keys()):
        miner_data_map[uid_key] = sorted(
            miner_data_map[uid_key], key=lambda x: x["time"]
        )

    # print(f"DEBUG: Returning for {field_name}: {dict(miner_data_map)}")
    return dict(miner_data_map)


# Helper to calculate aggregate time-series (e.g., average, max) from per-miner series
def calculate_aggregate_timeseries(
    all_miner_series: dict, aggregation_type: str = "average"
):
    """
    Calculates an aggregate time-series (average, max, sum) from a dictionary of per-miner time-series.
    all_miner_series: { "miner_uid1": [{"time": t, "value": v}, ...], ... }
    Returns: list of {"time": t, "value": aggregated_v}
    """
    if not all_miner_series:
        return []

    # Collect all data points by timestamp
    points_by_time = defaultdict(list)
    for miner_uid, series in all_miner_series.items():
        for point in series:
            points_by_time[point["time"]].append(point["value"])

    aggregated_series = []
    for time_str, values in sorted(points_by_time.items()):
        if not values:
            continue

        agg_value = None
        if aggregation_type == "average":
            agg_value = sum(values) / len(values)
        elif aggregation_type == "max":
            agg_value = max(values)
        elif aggregation_type == "sum":
            agg_value = sum(values)
        # Add other aggregations if needed

        if agg_value is not None:
            aggregated_series.append({"time": time_str, "value": agg_value})

    return sorted(aggregated_series, key=lambda x: x["time"])


@app.get("/metrics/global")
async def get_global_metrics(time_range: str = "1h"):
    """
    Derive all global overview metrics from individual miner data.
    """
    cache_key = f"derived_global_overview:{time_range}"
    cached = redis_client.get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            print(f"Warning: Malformed cache for key {cache_key}")

    final_result = {
        "all_miner_losses": {},
        "all_miner_perplexities": {},
        "global_average_loss_series": [],
        "global_average_perplexity_series": [],
        "global_max_epoch_series": [],
        "global_average_training_rate_series": [],
        "global_total_bandwidth_series": [],  # Or average, depending on what "bandwidth" field means
        "active_miners_count_series": [],
        "active_miners_current": 0,
    }

    try:
        # 1. Get all miner losses from "training_metrics"
        all_miner_losses = get_all_miner_timeseries_for_field(
            "training_metrics", "loss", time_range, query_api, INFLUXDB_BUCKET
        )
        final_result["all_miner_losses"] = all_miner_losses
        final_result["global_average_loss_series"] = calculate_aggregate_timeseries(
            all_miner_losses, "average"
        )

        # 2. Calculate perplexities from losses for each miner
        all_miner_perplexities = {}
        for miner_uid, loss_series in all_miner_losses.items():
            perplexity_series = []
            for point in loss_series:
                try:
                    if point.get("value") is not None:
                        perplexity_series.append(
                            {
                                "time": point["time"],
                                "value": math.exp(float(point["value"])),
                            }
                        )
                except (TypeError, ValueError):
                    pass  # Ignore errors for individual points
            all_miner_perplexities[miner_uid] = perplexity_series
        final_result["all_miner_perplexities"] = all_miner_perplexities
        final_result["global_average_perplexity_series"] = (
            calculate_aggregate_timeseries(all_miner_perplexities, "average")
        )

        # 3. Get all miner epochs from "training_metrics" and find max epoch series
        all_miner_epochs = get_all_miner_timeseries_for_field(
            "training_metrics", "epoch", time_range, query_api, INFLUXDB_BUCKET
        )
        final_result["global_max_epoch_series"] = calculate_aggregate_timeseries(
            all_miner_epochs, "max"
        )

        # 4. Get all miner training rates (e.g., "samples_per_second") from "training_metrics" and average
        #    Assuming 'samples_per_second' is the field for training rate. Adjust if different.
        all_miner_training_rates = get_all_miner_timeseries_for_field(
            "training_metrics",
            "samples_per_second",
            time_range,
            query_api,
            INFLUXDB_BUCKET,
        )
        final_result["global_average_training_rate_series"] = (
            calculate_aggregate_timeseries(all_miner_training_rates, "average")
        )

        # 5. Get all miner bandwidth from "network_metrics" and sum (or average)
        #    Assuming 'bandwidth' is the field name in 'network_metrics'.
        all_miner_bandwidths = get_all_miner_timeseries_for_field(
            "network_metrics", "bandwidth", time_range, query_api, INFLUXDB_BUCKET
        )
        # Summing bandwidth might make sense if it's individual usage. Averaging if it's a shared resource measure.
        final_result["global_total_bandwidth_series"] = calculate_aggregate_timeseries(
            all_miner_bandwidths, "sum"
        )

        # 6. Derive active_miners_count_series and current active miners (from loss data, for example)
        active_miner_timestamps = defaultdict(set)
        unique_miners_overall = set()
        for (
            miner_uid,
            series_data,
        ) in all_miner_losses.items():  # Use any comprehensive per-miner series
            unique_miners_overall.add(miner_uid)
            for point in series_data:
                active_miner_timestamps[point["time"]].add(miner_uid)

        active_count_series = []
        if active_miner_timestamps:  # Check if not empty
            for ts, uids in sorted(active_miner_timestamps.items()):
                active_count_series.append({"time": ts, "value": len(uids)})
            final_result["active_miners_current"] = len(
                active_miner_timestamps.get(
                    max(active_miner_timestamps.keys(), default=None), set()
                )
            )
        else:  # No data points found
            final_result["active_miners_current"] = 0

        final_result["active_miners_count_series"] = active_count_series

        redis_client.setex(cache_key, CACHE_TTL, json.dumps(final_result))
        return final_result
    except Exception as e:
        import traceback

        traceback.print_exc()
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
    """Get detailed AllReduce metrics including validator reports"""
    cache_key = f"allreduce_detailed:{time_range}"  # Use a new cache key
    cached = redis_client.get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            print(f"Warning: Malformed cache for key {cache_key}")

    # Query needs to ensure validator_uid is selected (Flux usually includes all tags)
    # The grouping in Flux might not be necessary anymore as we process in Python
    query = f''' 
    from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r._measurement == "allreduce_operations")
        |> keep(columns: ["_time", "_field", "_value", "operation_id", "epoch", "validator_uid"]) 
        |> group() // Ungroup might be simpler for Python processing
    '''
    try:
        result = query_api.query(query)
        # Use the new processing function
        processed_data = process_allreduce_validator_reports(result)

        # Cache the result
        redis_client.setex(cache_key, CACHE_TTL, json.dumps(processed_data))
        return processed_data
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching/processing AllReduce data: {str(e)}",
        )


@app.get("/locations/miners")
async def get_miner_locations():
    """Get the latest known geographic locations of active miners."""
    cache_key = "miner_locations_latest"
    cached = redis_client.get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            print(f"Warning: Malformed cache for key {cache_key}")

    locations = []
    # Get UIDs of miners considered 'active' (e.g., recently reported or present in metagraph)
    # Option 1: Use the existing /metrics/miners logic (queries InfluxDB for recent metagraph_metrics)
    # miner_uids_str = await get_miners_list() # Reuse existing endpoint logic
    # miner_uids = [int(uid_str) for uid_str in miner_uids_str if uid_str.isdigit()]

    # Option 2: Use UIDs known to have location data in Redis (simpler if updater runs reliably)
    miner_location_keys = redis_client.keys("miner_location:*")
    miner_uids = [int(key.decode().split(":")[1]) for key in miner_location_keys]

    print(f"Retrieving locations for {len(miner_uids)} UIDs found in Redis keys.")

    for uid in miner_uids:
        redis_key = f"miner_location:{uid}"
        location_data = redis_client.hgetall(redis_key)
        if location_data:
            try:
                # Decode from bytes and convert types
                locations.append(
                    {
                        "uid": int(
                            location_data.get(b"uid", uid)
                        ),  # Fallback to loop uid
                        "lat": float(location_data.get(b"lat", 0.0)),
                        "lon": float(location_data.get(b"lon", 0.0)),
                        "city": location_data.get(b"city", b"Unknown").decode(),
                        "country": location_data.get(b"country", b"Unknown").decode(),
                        "last_updated": location_data.get(
                            b"last_updated", b""
                        ).decode(),
                    }
                )
            except (ValueError, TypeError, KeyError) as e:
                print(
                    f"Error processing location data for UID {uid} from Redis: {e} - Data: {location_data}"
                )

    # Cache the result
    redis_client.setex(cache_key, CACHE_TTL, json.dumps(locations))
    print(f"Returning {len(locations)} miner locations.")
    return locations


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


def process_allreduce_validator_reports(result):
    """
    Process AllReduce metrics, grouping by operation and nesting validator reports.
    Output: [
        {
            "operation_id": "...",
            "epoch": "...",
            "representative_time": "...", // e.g., earliest time reported for this op
            "validator_reports": [
                {"validator_uid": "...", "time": "...", "metrics": {...}},
                ...
            ]
        }, ...
    ]
    """
    # Use defaultdict for easier handling of nested structures
    operations_data = defaultdict(lambda: {"validator_reports": [], "times": []})

    for table in result:
        for record in table.records:
            try:
                op_id = record.values.get("operation_id")
                epoch = record.values.get("epoch")
                validator_uid = record.values.get("validator_uid")
                field = record.values.get("_field")
                value = record.values.get("_value")
                time = record.values.get("_time")  # Keep the specific report time

                if not all([op_id, epoch, validator_uid, field, time]):
                    # Skip incomplete records
                    continue

                # Use (op_id, epoch) as the main key
                op_key = (op_id, epoch)

                # Find or create the report for this validator within this operation
                validator_report = None
                for report in operations_data[op_key]["validator_reports"]:
                    if report["validator_uid"] == validator_uid:
                        validator_report = report
                        break

                if validator_report is None:
                    validator_report = {
                        "validator_uid": validator_uid,
                        "time": time.isoformat()
                        if isinstance(time, datetime)
                        else str(time),
                        "metrics": {},
                    }
                    operations_data[op_key]["validator_reports"].append(
                        validator_report
                    )
                    # Store time for finding representative time later
                    operations_data[op_key]["times"].append(time)

                # Add the metric to this validator's report
                validator_report["metrics"][field] = value

            except Exception as e:
                # Log potential errors during record processing
                print(f"Error processing record: {record.values} - {e}")
                continue

    # Convert the defaultdict to the desired list structure
    processed_list = []
    for (op_id, epoch), data in operations_data.items():
        if not data["validator_reports"]:  # Skip if no valid reports were added
            continue

        # Sort validator reports for consistency (e.g., by UID)
        data["validator_reports"].sort(key=lambda x: int(x["validator_uid"]))

        # Determine a representative time (e.g., the earliest report time)
        representative_time = min(data["times"]) if data["times"] else None

        processed_list.append(
            {
                "operation_id": op_id,
                "epoch": epoch,
                "representative_time": representative_time.isoformat()
                if isinstance(representative_time, datetime)
                else str(representative_time),
                "validator_reports": data["validator_reports"],
            }
        )

    # Sort the final list of operations (e.g., by time descending)
    processed_list.sort(key=lambda x: x["representative_time"], reverse=True)

    return processed_list


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
