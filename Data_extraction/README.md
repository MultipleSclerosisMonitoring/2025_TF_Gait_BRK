# Data Extraction

**Overview**

The Data Extraction Project is designed to process an input Excel file and extract data chunks based on specified durations (e.g., 10 or 15 seconds). The script focuses only on the relevant columns `Reference`, `datefrom`, `dateuntil`, and `mov_type` dividing the given timeline into fixed-duration segments. For each reference, the extraction is performed for both legs (Left and Right), and each chunk is saved as an individual output file with a structured naming convention. Additionally, the extraction process prints out a clear summary of the processed data, which may include the total extracted durations and sample counts for both walking and non-walking segments.

**Features**
1. Selective Column Processing The script reads only the columns: Reference, datefrom, dateuntil, and mov_type, ignoring any extra columns in the Excel file.
2. Chunk Extraction The timeline between datefrom and dateuntil is divided into chunks of a specified duration (e.g., 10 or 15 seconds). Any remaining time at the
   end that doesn't meet the complete duration is discarded.
3. Dual Extraction For every reference, data is extracted for both Left and Right legs.
4. Output File Naming Each chunk is saved as a separate `.xlsx` file using the naming convention: <mov_type>+<start_time>+<end_time>+<leg>.xlsx</br>
   Example:
   `walking+2024-04-24_09-48-08+2024-04-24_09-48-13+Left.xlsx`
   `not_walking+2024-04-24_11-00-10+2024-04-24_11-00-15+Right.xlsx`
5. Extraction Summary After processing, the script outputs a summary with the total number of references and can be extended to include detailed statistics, such
    as the overall duration and number of samples extracted for both walking and non-walking segments.

**Usage**<br/>
Run the script using the command-line interface:
```
python scripts/data_extraction.py -i input_data/sample_input.xlsx -o output_data/ -d 10 -p config/config_db.yaml
```
   Arguments:<br/>
      - `-i` or `--input`: Path to the input Excel file.<br/>
      - `-o` or `--output`: Path to the directory where chunk files should be saved.<br/>
      - `-d` or `--duration`: Chunk duration in seconds (e.g., `10` or `15`).<br/>
      - `-p` or `--path`: Path to the configuration file for `cInfluxDB`.<br/>

**Extraction Summary Tip**

At the end of the extraction process, the script outputs a summary that includes:
- The total number of references processed.
- The number of chunks (and potentially total duration) successfully extracted for walking segments.
- The number of chunks (and potentially total duration) successfully extracted for non-walking segments.

This summary aids in quickly assessing the volume of data processed and provides insights into any data segments that might have been skipped due to insufficient duration.

**Error Handling**
1. Invalid or missing columns in the input file will raise an error.
2. Extra seconds not fitting into the chunk duration will be discarded.
3. Handles issues with file paths, directories, and date conversions.
