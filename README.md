# Kuopio Full-Body Processing

These scripts preform the processing steps for the Kuopio Full-Body Dataset.
They transform the raw Vicon and Xsens data into formats usable
in OpenSim and other processing programs.

The repository contains processing scripts in Python and C++ (for OpenSim).
The following sections describe how to setup and run the code.

## Python Scripts

The Python code does not require importing OpenSim. All dependencies are defined
in the `pyproject.toml` file and locked to their minor version.

### Environment Setup
The Python code was tested against `Python 3.12`

To configure the environment:

1. Setup a Python 3.12 virtual environment: `python -m venv .venv` or `uv venv`.
If you are using `uv` it will automatically detect and create the correct Python version.
If you are using `python -m venv` you need to ensure the Python you invoke is 3.12.
The code should also work with other versions of Python but this hasn't been tested.

2. Activate the venv: `source .venv/bin/active`. This must be done EVERY time
you start to work on the project from a new terminal.

3. Install the project dependencies in dev mode: `pip install -e '.[dev]'` or `uv pip install -e '.[dev]'`

## C++ Scripts (OpenSim)
The C++ OpenSim processing scripts are contained in the folder `opensim_cpp`.
Each project has its own subdirectory. A Docker container with the OpenSim 4.6 release
is provided to enable development and reproduction on any OS with Docker support.

### Environment Setup (Docker)
To setup the Docker container:

1. [Install Docker](https://docs.docker.com/get-started/get-docker/) and start Docker.
2. Build the development container (this may take some time depending on your machine's power):
```bash
docker build . -t kfb-opensim-cpp
```
3. Start the development container. You will now have a shell into the container
where the OpenSim dependencies and environment is configured:
```bash
docker run -it -v ~/data:/root/data -v ./opensim_cpp:/root/opensim_cpp kfb-opensim-cpp
```
*NOTE*: The Docker volume mount syntax is
`-v <host_path>:<container_path>`. The `~/data:/root/data` mount is so you can process
data within the container that will be reflected on your local machine. You may need to adjust this
mount depending on where you store the data you would like to process.
The `./opensim_cpp:/root/opensim_cpp` mount allows the directory containing the programs to be linked to the container so when you make code changes you do not need to rebuild the container.

## Data Processing
To begin, you must define the desired input and output base directories for processing:

```bash
export INPUT_PATH=~/data/kuopio-full-body-dataset
export OUTPUT_PATH=~/data/kuopio-full-body-dataset
```

Both directories can be the same, if you would like the processing results to be
output to the same directory as the input data (in different sub-directories).

The following processing stages require both the Docker C++ scripts and the Python
scripts so ensure you have both environments configured as above before continuing.

### C3D Extraction
This stage extracts the raw data from the manufacturer's C3D binary format
into human readable files.

To run extraction:

1. Run the Docker container described above.
2. From within the Docker container, ensure you define `INPUT_PATH` and `OUTPUT_PATH`
3. Build the `C3DParserBulk` project:
```bash
cd C3DParserBulk
cmake . -B build
cd build
make -j
```
4. Run the C3D extraction code. Remember to recompile if you make code changes:
```bash
./main $INPUT_PATH/s01_raw $OUTPUT_PATH/s02_extracted
```

### XSens MTB Extraction
This stage extracts the raw data from the manufacturer's MTB binary format
into human readable files.

To run extraction:

1. Setup the [gateway240/BatchExportMTB](https://github.com/gateway240/BatchExportMTB) repository in a new environment.
2. Run the `.mtb` export script after setting up the repository.
3. From the Python environment in this project, check sensor uniformity after export (fix if necessary by removing `--dry-run`):
```bash
python src/kfb_processing/check_imu_length.py $OUTPUT_PATH/s02_extracted --dry-run
```
The OpenSim parser cannot handle non-uniform trial lengths. This script will trim
all sensors to the same length using the sensor with the least amount of frames.
4. From the Docker container, build the `IMUXsensBulkV2` project:
```bash
cd IMUXsensBulkV2
cmake . -B build
cd build
make -j
```
5. Run the XSens consolidation code (ensuring `OUTPUT_PATH` is defined):
```bash
./main $OUTPUT_PATH/s02_extracted $OUTPUT_PATH/s02_extracted
```
6. From the Python environment, check the extraction results:
```bash
python src/kfb_processing/check_dataset.py $OUTPUT_PATH/s02_extracted
```

### Alignment
The following C++ script aligns the IMU and Vicon signals based on the analog trigger
signal.

To run processing:

1. From the Docker container, build the `SignalAlign` project:
```bash
 cd SignalAlign
 cmake . -B build
 cd build
 make -j
```
2. Run the project (ensuring `OUTPUT_PATH` is defined):
```bash
./main "$OUTPUT_PATH/s02_extracted" "$OUTPUT_PATH/s03_aligned" 2>&1 | tee "output_$(date +%Y%m%d_%H%M%S).txt"
```
Investigate if match doesn't exist (e.g. IMU wasn't recorded).


### Technical Validation
The following scripts are run from the Python environment to perform the technical
validation experiments.

Check results:
```bash
python src/kfb_processing/check_dataset.py $OUTPUT_PATH/s03_aligned
python src/kfb_processing/check_optical_data.py $OUTPUT_PATH/s03_aligned --output_dir $OUTPUT_PATH/technical_validation
python src/kfb_processing/check_imu_table_test.py $OUTPUT_PATH/s02_extracted --output_dir $OUTPUT_PATH/technical_validation
python src/kfb_processing/check_imu_continuity.py $OUTPUT_PATH/s02_extracted --output_dir $OUTPUT_PATH/technical_validation
python src/kfb_processing/check_imu_marker_correlation.py $OUTPUT_PATH/s03_aligned/ --output_dir $OUTPUT_PATH/technical_validation
python src/kfb_processing/check_imu_marker_correlation_summary.py $OUTPUT_PATH/s03_aligned/ --output_dir $OUTPUT_PATH/technical_validation
python src/kfb_processing/check_analog.py $OUTPUT_PATH/s03_aligned/ --output_dir $OUTPUT_PATH/technical_validation
python src/kfb_processing/check_analog_summary.py $OUTPUT_PATH/s03_aligned/ --output_dir $OUTPUT_PATH/technical_validation
```

## For Developers
The following code is a useful reference for developers and not necessary for
most users.

Debugging
```bash
python src/kfb_processing/check_analog.py $OUTPUT_PATH/s02_extracted/ --output_dir $OUTPUT_PATH/emg/_output
```

### Dataset readme generation

```py
python src/kfb_processing/demographic_info.py --output_dir ./out --input_dir $OUTPUT_PATH
# This format works for the zenodo online preview
python src/kfb_processing/generate_readme.py --output_dir $OUTPUT_PATH --table_fmt github
python src/kfb_processing/generate_readme.py --output_dir out --doc_fmt html

python src/kfb_processing/trial_info.py --output_dir out --input_csv movements.csv --input_dir data/measurement-config

python src/kfb_processing/emg_info.py --output_dir out --input_csv emg-sensor-mappings.csv --input_dir data/measurement-config
```

### Zip result
```bash
cd $OUTPUT_PATH
7z a -tzip -mmt=on s01_raw.zip ./s01_raw
7z a -tzip -mmt=on s02_extracted.zip ./s02_extracted
7z a -tzip -mmt=on s03_aligned.zip ./s03_aligned
7z a -tzip -mmt=on technical_validation.zip ./technical_validation

7z a -tzip -mmt=on kuopio-full-body-dataset.zip ./kuopio-full-body-dataset/
```

### Graph Directory Structure

```bash
python src/kfb_processing/dir_to_graph.py $OUTPUT_PATH -o images/dataset_graph.d2 --font-size 54
d2 images/dataset_graph.d2 images/dataset_graph.png
```
