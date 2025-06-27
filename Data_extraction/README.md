# Data Extraction

**Overview**

The Data Extraction script processes an input Excel file containing movement metadata and pulls matching time-series data from InfluxDB. It divides each record’s timeline into fixed-duration chunks (e.g., 5s or 10 s), queries both “Left” and “Right” legs for each chunk, writes each result to its own `.xlsx` file, and prints a detailed summary of extraction statistics.

---

## Features

1. **Modular Design**  
   - **DataLoader:** Reads only the required columns (`Reference`, `datefrom`, `dateuntil`, `mov_type`) and converts date fields.  
   - **ChunkProcessor:** Splits each time interval into non-overlapping chunks of a user-specified duration.  
   - **DataExtractor:** Connects to InfluxDB (via `cInfluxDB`) and retrieves data for each chunk and leg.  
   - **SummaryManager:** Tracks and reports:
     - Total chunks fully extracted
     - Chunks discarded due to insufficient remaining duration
     - Cumulative duration and sample counts for walking and non-walking segments  

2. **Flexible Chunking**  
   Any leftover time that doesn’t fill a complete chunk is automatically discarded and counted in the summary.

3. **Dual-Leg Extraction**  
   For each time chunk, data is pulled separately for both `Left` and `Right` legs.

4. **Structured Output**  
   Each chunk is saved as a separate Excel file named:  
   ```
   <Reference>+<mov_type>+<start_time>+<end_time>+<leg>.xlsx
   ```
   Example:  
   `ABC123+walking+2024-04-24_09-48-08+2024-04-24_09-48-18+Left.xlsx`

5. **Clear Extraction Summary**  
   At completion, the script prints:
   - Total references (i.e., chunks) fully extracted  
   - Total chunks skipped (insufficient duration)  
   - Walking vs. non-walking total duration and sample counts  

---

## Usage

```bash
python scripts/extract_data.py \
  -i input_data/sample_input.xlsx \
  -o output_data/ \
  -d 10 \
  -p config/config_db.yaml
```

### Arguments

- `-i`, `--input`  
  Path to the input Excel file containing `Reference`, `datefrom`, `dateuntil`, `mov_type`.
- `-o`, `--output`  
  Directory where chunk files will be saved. Will be created if it doesn’t exist.
- `-d`, `--duration`  
  Chunk duration in seconds (e.g., `10` or `15`).
- `-p`, `--path`  
  Path to the InfluxDB configuration file (YAML) for `cInfluxDB`.  
  **Default:** `InfluxDBms/config_db.yaml`

---

## Extraction Summary

After the run, you’ll see:

- **Total references extracted:** count of chunks fully processed  
- **Total references not extracted:** chunks skipped due to leftover time  
- **Walking segments:** total seconds extracted & sample count  
- **Non-walking segments:** total seconds extracted & sample count  

This quick overview helps verify data volume and identify any gaps in your input.

---

## Error Handling

1. **Missing or Invalid Columns**  
   An error is raised if the input file lacks any of `Reference`, `datefrom`, `dateuntil`, or `mov_type`.

2. **InfluxDB Connection Issues**  
   Failures to initialize or query `cInfluxDB` are caught and reported per chunk/leg.

3. **File I/O Errors**  
   Any problem writing the output `.xlsx` file is logged, and processing continues for remaining chunks.

4. **Chunk Duration Mismatch**  
   Leftover seconds not meeting the full chunk duration are tracked and reported, but not written.

---  
