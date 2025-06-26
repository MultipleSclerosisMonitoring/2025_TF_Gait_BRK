import os
import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd
from dateutil.parser import parse as parse_date
from InfluxDBms.cInfluxDB import cInfluxDB


class DataLoader:
    """Handles reading and preprocessing of the input Excel file.

    Attributes:
        input_file (str): Path to the input Excel file.
        df (pd.DataFrame): DataFrame containing loaded data.
    """
    
    def __init__(self, input_file: str):
        """
        Initializes the DataLoader with the provided file path.

        Args:
            input_file (str): Path to the input Excel file.
        """
        self.input_file = input_file
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the Excel file and converts date columns.

        The file is expected to contain the following columns:
        'Reference', 'datefrom', 'dateuntil', and 'mov_type'.

        Returns:
            pd.DataFrame: DataFrame with the selected columns and properly
            converted date fields.

        Raises:
            Exception: An error occurred while reading or processing the file.
        """
        try:
            self.df = pd.read_excel(self.input_file)
            # Select only the required columns.
            self.df = self.df[['Reference', 'datefrom', 'dateuntil', 'mov_type']]
            # Convert the date columns to datetime objects.
            self.df['datefrom'] = pd.to_datetime(self.df['datefrom'])
            self.df['dateuntil'] = pd.to_datetime(self.df['dateuntil'])
        except Exception as e:
            print(f"Error reading or processing input file: {e}")
        return self.df


class ChunkProcessor:
    """Creates time chunks for each row of data based on a fixed duration.

    Attributes:
        chunk_duration (timedelta): The duration of each data chunk.
    """
    
    def __init__(self, chunk_duration: int):
        """
        Initializes the ChunkProcessor with the chunk duration in seconds.

        Args:
            chunk_duration (int): Duration in seconds for each chunk.
        """
        self.chunk_duration = timedelta(seconds=chunk_duration)

    def create_chunks(self, row: pd.Series) -> list:
        """
        Generates a list of time chunk tuples for a given row of data.

        Each chunk is represented as a tuple (start_time, end_time), generated
        by stepping through the time window defined by 'datefrom' and 'dateuntil'.

        Args:
            row (pd.Series): A row from the DataFrame with 'datefrom' and 'dateuntil' columns.

        Returns:
            list: List of tuples, where each tuple contains (start_time, end_time) for a chunk.
        """
        chunks = []
        current_time = row['datefrom']
        while current_time + self.chunk_duration <= row['dateuntil']:
            chunk_end_time = current_time + self.chunk_duration
            chunks.append((current_time, chunk_end_time))
            current_time = chunk_end_time
        return chunks


class DataExtractor:
    """Retrieves data from InfluxDB for provided time chunks.

    Attributes:
        config_path (str): Path to the configuration file.
        influx_db (cInfluxDB): Instance of the InfluxDB client.
    """
    
    def __init__(self, config_path: str):
        """
        Initializes the DataExtractor with the configuration required for InfluxDB.

        Args:
            config_path (str): Path to the configuration file for InfluxDB.
        """
        self.config_path = config_path
        try:
            self.influx_db = cInfluxDB(config_path=config_path)
        except Exception as e:
            print(f"Error initializing cInfluxDB: {e}")
            self.influx_db = None

    def extract_data(self, from_time: datetime, until_time: datetime, reference: str, leg: str) -> pd.DataFrame:
        """
        Extracts data from InfluxDB using the given time interval and parameters.

        This method queries the InfluxDB for data between from_time and until_time
        and prints the dataset size. The data is sorted by the '_time' column after
        converting it to GMT+1 and removing the timezone info.

        Args:
            from_time (datetime): The start time of the data chunk.
            until_time (datetime): The end time of the data chunk.
            reference (str): The reference token used for querying.
            leg (str): Specifies the leg ("Left" or "Right") for the extraction.

        Returns:
            pd.DataFrame: A sorted DataFrame containing the query results.
                         Returns None if the query fails.
        """
        if self.influx_db is None:
            print("InfluxDB is not initialized.")
            return None

        try:
            df = self.influx_db.query_data(from_time, until_time, qtok=reference, pie=leg)
            # Adjust _time column to GMT+1 and remove timezone info.
            gmt_plus_1_fixed = timezone(timedelta(hours=1))
            df['_time'] = df['_time'].dt.tz_convert(gmt_plus_1_fixed).dt.tz_localize(None)
            df_sorted = df.sort_values(by="_time", ascending=False)
            print(f"Results of the query: Dataset size {df_sorted.shape}")
            return df_sorted
        except Exception as e:
            print(f"Error querying data for {reference} ({leg}): {e}")
            return None


class SummaryManager:
    """Tracks and prints summary statistics for the extraction process.

    Attributes:
        extracted_count (int): Number of fully extracted chunks.
        not_extracted_count (int): Number of chunks that didn't meet the full duration.
        walking_duration (int): Total duration (in seconds) extracted for walking segments.
        walking_samples (int): Count of walking samples.
        non_walking_duration (int): Total duration (in seconds) extracted for non-walking segments.
        non_walking_samples (int): Count of non-walking samples.
    """
    
    def __init__(self):
        """Initializes all summary counters to zero."""
        self.extracted_count = 0
        self.not_extracted_count = 0
        self.walking_duration = 0
        self.walking_samples = 0
        self.non_walking_duration = 0
        self.non_walking_samples = 0

    def update_extraction(self, movement_type: str, chunk_duration: int):
        """
        Updates extraction statistics based on the movement type.

        Args:
            movement_type (str): The movement type (e.g., 'walking').
            chunk_duration (int): The duration (in seconds) of each chunk.
        """
        if str(movement_type).lower() == 'walking':
            self.walking_duration += chunk_duration
            self.walking_samples += 1
        else:
            self.non_walking_duration += chunk_duration
            self.non_walking_samples += 1

    def add_extracted_chunk(self):
        """Increments the count of fully extracted chunks."""
        self.extracted_count += 1

    def add_not_extracted_chunk(self):
        """Increments the count of chunks that were not fully extracted."""
        self.not_extracted_count += 1

    def print_summary(self):
        """
        Prints the overall summary of extraction statistics including:
          - Number of fully extracted chunks.
          - Number of chunks not meeting the full duration.
          - Total extracted durations and sample counts for walking and non-walking segments.
        """
        print("Extraction Summary:")
        print(f"Total references (chunks) extracted: {self.extracted_count}")
        print(f"Total references (chunks) not extracted (insufficient duration): {self.not_extracted_count}")
        print(f"For walking segments: Total duration extracted = {self.walking_duration} seconds, "
              f"Total samples = {self.walking_samples}")
        print(f"For non-walking segments: Total duration extracted = {self.non_walking_duration} seconds, "
              f"Total samples = {self.non_walking_samples}")


def main():
    """
    Main function to orchestrate the data extraction process.

    This function handles:
      - Parsing command-line arguments.
      - Loading data using DataLoader.
      - Generating time chunks using ChunkProcessor.
      - Extracting data via DataExtractor.
      - Updating and printing extraction summaries via SummaryManager.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Execution of Data Extraction with modular design.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input Excel file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('-d', '--duration', type=int, required=True, help='Chunk duration in seconds')
    parser.add_argument('-p', '--path', type=str, default='InfluxDBms/config_db.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    # Load the data.
    loader = DataLoader(args.input)
    df = loader.load_data()
    if df is None:
        return

    # Ensure output directory exists.
    os.makedirs(args.output, exist_ok=True)

    # Instantiate components.
    chunk_processor = ChunkProcessor(args.duration)
    extractor = DataExtractor(args.path)
    summary_manager = SummaryManager()

    # Process each row in the data.
    for index, row in df.iterrows():
        # Generate time chunks for the current row.
        chunks = chunk_processor.create_chunks(row)
        reference = row['Reference']
        mov_type = row['mov_type']

        # Process each chunk.
        for (start_time, end_time) in chunks:
            # Extract for both "Left" and "Right" legs.
            for leg in ['Left', 'Right']:
                filename = (
                    f"{reference}+{mov_type}+{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
                    f"+{end_time.strftime('%Y-%m-%d_%H-%M-%S')}+{leg}.xlsx"
                )
                output_file = os.path.join(args.output, filename)
                df_extracted = extractor.extract_data(start_time, end_time, reference, leg)
                if df_extracted is not None:
                    try:
                        df_extracted.to_excel(output_file)
                    except Exception as e:
                        print(f"Error writing file {output_file}: {e}")
                # Update summary for each leg extraction.
                summary_manager.update_extraction(mov_type, args.duration)
            # Mark the chunk as fully extracted (both legs processed).
            summary_manager.add_extracted_chunk()

        # If leftover time remains that doesn't form a complete chunk, count it as not extracted.
        total_possible = (row['dateuntil'] - row['datefrom']).total_seconds()
        if total_possible > (len(chunks) * args.duration):
            summary_manager.add_not_extracted_chunk()

    # Print the final extraction summary.
    summary_manager.print_summary()


if __name__ == "__main__":
    main()
