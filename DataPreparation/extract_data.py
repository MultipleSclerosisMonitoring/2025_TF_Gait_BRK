import os
import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd
from dateutil.parser import parse as parse_date
from InfluxDBms.cInfluxDB import cInfluxDB

# Function to convert strings to datetime objects
def parse_datetime(value):
    try:
        return parse_date(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}. Error: {e}")

# Get default dates
def get_default_dates():
    now = datetime.now(timezone.utc)
    return now.isoformat() + "Z", (now + timedelta(minutes=30)).isoformat() + "Z"

# Function to create chunks and run extraction
def create_chunks_and_extract(df, output_dir, chunk_duration, config_path):
    chunk_duration_td = timedelta(seconds=chunk_duration)
    extracted_count = 0
    not_extracted_count = 0

    for index, row in df.iterrows():
        datefrom = row['datefrom']
        dateuntil = row['dateuntil']
        mov_type = row['mov_type']

        current_time = datefrom
        while current_time + chunk_duration_td <= dateuntil:
            chunk_end_time = current_time + chunk_duration_td
            for leg in ['Left', 'Right']:
                chunk_file = os.path.join(
                    output_dir,
                    f"{mov_type}+{current_time.strftime('%Y-%m-%d_%H-%M-%S')}+{chunk_end_time.strftime('%Y-%m-%d_%H-%M-%S')}+{leg}.xlsx"
                )
                run_extraction(current_time, chunk_end_time, row['Reference'], leg, config_path, chunk_file)
            current_time = chunk_end_time
            extracted_count += 1

        # If extra seconds are not long enough for the chunk, do not extract
        if current_time < dateuntil:
            not_extracted_count += 1

    print(f"Total references extracted: {extracted_count}")
    print(f"Total references not extracted: {not_extracted_count}")

# Function to run the extraction tool
def run_extraction(from_time, until_time, qtok, leg, config_path, output_file):
    # Ensure the input times are proper datetime objects
    if not isinstance(from_time, datetime):
        print(f"Error: from_time is not a datetime object: {from_time}")
        return
    if not isinstance(until_time, datetime):
        print(f"Error: until_time is not a datetime object: {until_time}")
        return

    from_time_str = from_time.isoformat() + "Z"
    until_time_str = until_time.isoformat() + "Z"

    try:
        iDB = cInfluxDB(config_path=config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error initializing cInfluxDB: {e}")
        return

    try:
        df = iDB.query_data(from_time, until_time, qtok=qtok, pie=leg)
        gmt_plus_1_fixed = timezone(timedelta(hours=1))
        df['_time'] = df['_time'].dt.tz_convert(gmt_plus_1_fixed).dt.tz_localize(None)
        print(f"Results of the query: Dataset size {df.shape}")
        df_sorted = df.sort_values(by="_time", ascending=False)
        df_sorted.to_excel(output_file)
    except Exception as e:
        print(f"Error querying data: {e}")
        return

# Main function
def main():
    parser = argparse.ArgumentParser(description='Execution of Data Extraction.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input Excel file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('-d', '--duration', type=int, required=True, help='Chunk duration in seconds')
    parser.add_argument('-p', '--path', type=str, default='InfluxDBms/config_db.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    # Load the input Excel file
    try:
        df = pd.read_excel(args.input)
        df = df[['Reference', 'datefrom', 'dateuntil', 'mov_type']]  # Select only required columns
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Ensure correct data types
    try:
        df['datefrom'] = pd.to_datetime(df['datefrom'])
        df['dateuntil'] = pd.to_datetime(df['dateuntil'])
    except Exception as e:
        print(f"Error processing date columns: {e}")
        return

    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Create chunks and run extraction
    create_chunks_and_extract(df, args.output, args.duration, args.path)

if __name__ == "__main__":
    main()
