# Kuopio Full Body Processing

These scripts transform the raw Vicon and Xsens data into formats usable
in OpenSim. 

## Python processing

Setting the export path and configuring the env (`python -m venv .venv` or `uv venv`):

```py
export OUTPUT_PATH=~/data/kuopio-full-body-dataset
export OUTPUT_PATH=out/kuopio-full-body-dataset
source .venv/bin/activate
uv pip install -e .[dev]
```

### Dataset readme generation

```py
python src/demographic-info.py --output_dir ./out --input_dir $OUTPUT_PATH
# This format works for the zenodo online preview
python src/generate-readme.py --output_dir $OUTPUT_PATH --table_fmt github
python src/generate-readme.py --output_dir out --doc_fmt html
```

## OpenSim
For autocomplete to work:
```bash
 cd C3DParserBulk
 cmake . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=on
 cd build
 make -j$(nproc)
./main $OUTPUT_PATH/s01_raw $OUTPUT_PATH/s02_extracted
```

## Consolidate IMU Xsens
See https://github.com/gateway240/BatchExportMTB for extracting `.mtb` data from Xsens IMUs

Check sensor uniformity (fix if necessary by removing `--dry-run`):
```bash
python src/check_imu_length.py $OUTPUT_PATH/s02_extracted --dry-run
```
Consolidate IMUs:
```bash
 cd IMUXsensBulkV2
 cmake . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=on
 cd build
 make -j$(nproc)
./main $OUTPUT_PATH/s02_extracted $OUTPUT_PATH/s02_extracted
```
Check results:
```bash
python src/check_dataset.py $OUTPUT_PATH/s02_extracted
```

## Align Signal
```bash
 cd SignalAlign
 cmake . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=on
 cd build
 make -j$(nproc)
./main $OUTPUT_PATH/s02_extracted $OUTPUT_PATH/s03_aligned
```
Add files manually if match doesn't exist (e.g. IMU wasn't recorded)

Check results:
```bash
python src/check_dataset.py $OUTPUT_PATH/s03_aligned
python src/check_optical_data.py $OUTPUT_PATH/s03_aligned --output_dir $OUTPUT_PATH
python src/check_imu_table_test.py $OUTPUT_PATH/imu_table_test_extracted --output_dir $OUTPUT_PATH/imu_table_test_extracted/_output
python src/check_imu_continuity.py $OUTPUT_PATH/s02_extracted --output_dir $OUTPUT_PATH/imu_table_test_extracted/_output
python src/check_imu_marker_correlation.py $OUTPUT_PATH/s02_extracted --output_dir $OUTPUT_PATH/untrimmed-correlation
python src/check_imu_marker_correlation.py $OUTPUT_PATH/s03_aligned/ --output_dir $OUTPUT_PATH/imu_table_test_extracted/_output
python src/check_emg.py $OUTPUT_PATH/s03_aligned/ --output_dir $OUTPUT_PATH/emg/_output
```

## Zip result
```bash
cd $OUTPUT_PATH
7z a -tzip -mmt=on kfb-s02_extracted.zip ./s02_extracted
7z a -tzip -mmt=on kfb-s03_aligned.zip ./s03_aligned

7z a -tzip -mmt=on kuopio-full-body-dataset.zip ./kuopio-full-body-dataset/
```
