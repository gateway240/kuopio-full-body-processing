/* -------------------------------------------------------------------------- *
 *                            OpenSim:  main.cpp                              *
 * -------------------------------------------------------------------------- *
 * The OpenSim API is a toolkit for musculoskeletal modeling and simulation.  *
 * See http://opensim.stanford.edu and the NOTICE file for more information.  *
 * OpenSim is developed at Stanford University and supported by the US        *
 * National Institutes of Health (U54 GM072970, R24 HD065690) and by DARPA    *
 * through the Warrior Web program.                                           *
 *                                                                            *
 * Copyright (c) 2005-2024 Stanford University and the Authors                *
 * Author(s): Alex Beattie, Ayman Habib, Ajay Seth                            *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may    *
 * not use this file except in compliance with the License. You may obtain a  *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.         *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 * -------------------------------------------------------------------------- */

// INCLUDES
#include <OpenSim/Common/C3DFileAdapter.h>
#include <OpenSim/Common/STOFileAdapter.h>
#include <OpenSim/Common/TRCFileAdapter.h>

#include <SimTKcommon/SmallMatrix.h>
#include <SimTKcommon/internal/Quaternion.h>
#include <chrono> // for std::chrono functions
#include <cstddef>
#include <filesystem>
#include <future> // For std::async, std::future
#include <iostream>
#include <string>

// const auto &MAX_THREADS = 12;
const auto &MAX_THREADS = std::thread::hardware_concurrency();

namespace fs = std::filesystem;

// Trim a TimeSeriesTable to a time window and write to new location
template <typename A, typename T>
std::pair<double, double> trimAndWrite(OpenSim::TimeSeriesTable_<T> &table,
                                       const fs::path &outFile, double tStart,
                                       double tEnd) {
  // std::cout << "Old Length: " << table.getIndependentColumn().size() <<
  // std::endl;

  const auto &timeCol = table.getIndependentColumn();

  const size_t &closest_start = table.getNearestRowIndexForTime(tStart);
  const size_t &closest_end = table.getNearestRowIndexForTime(tEnd, false);

  const size_t &before_start = table.getRowIndexBeforeTime(tStart);
  const size_t &before_end = table.getRowIndexBeforeTime(tEnd);

  const size_t &after_start = table.getRowIndexAfterTime(tStart);
  const size_t &after_end = table.getRowIndexAfterTime(tEnd);

  const size_t &start_index = before_start;
  const size_t &end_index = before_end;

  if (after_start - before_start != 0 || after_end - before_end != 0) {
    std::cout << "File: " << outFile << " Target start: " << tStart
              << " After start: " << timeCol[after_start] << " (+ "
              << timeCol[after_start] - tStart << ")"
              << " Closest time: " << timeCol[closest_start]
              << " Before time: " << timeCol[before_start] << " (- "
              << tStart - timeCol[before_start] << ")"
              << " start index " << start_index << " closest start "
              << closest_start << " before start " << before_start
              << " diff: " << after_start - before_start << std::endl;
    std::cout << "File: " << outFile << " Target end: " << tEnd
              << " After end: " << timeCol[after_end] << " (+ "
              << timeCol[after_end] - tEnd << ")"
              << " Closest time: " << timeCol[closest_end]
              << " Before time: " << timeCol[before_end] << " (- "
              << tEnd - timeCol[before_end] << ")"
              << " start index " << end_index << " closest start "
              << closest_end << " before start " << before_end
              << " diff: " << after_end - before_end << std::endl;
  }

  // do the actual trimming based on index instead of time.
  // CANNOT use table.trim because it uses the "next" time.
  // we need the closest time to avoid the off by 1 issue
  table.trimToIndices(start_index, end_index);
  const double t0 = table.getIndependentColumn().front();
  const auto &length = table.getNumRows();
  const auto &ind_col = table.getIndependentColumn();
  for (size_t i = 0; i < length; ++i) {
    table.setIndependentValueAtIndex(i, ind_col[i] - t0);
  }

  A::write(table, outFile.string());
  // std::cout << "Trimmed and saved: " << outFile << std::endl;
  // Get new time range
  const auto &newTimes = table.getIndependentColumn();
  if (newTimes.empty()) {
    std::cerr << "Warning: Trimmed table is empty." << std::endl;
    return {-1.0, -1.0};
  }
  double newStart = newTimes.front();
  double newEnd = newTimes.back();
  // std::cout << "New start: " << newStart << " New end: " << newEnd <<
  // std::endl;
  return {timeCol[start_index], timeCol[end_index]};
}

// Process a single trial
void processTrial(const fs::path &analogFile, const fs::path &originalRoot,
                  const fs::path &newRoot) {
  std::cout << "Processing: " << analogFile << std::endl;

  // Get trial name
  std::string trialName = analogFile.stem().string();
  if (trialName.find("_analog") == std::string::npos) {
    std::cerr << "Skipping non-analog file: " << analogFile << std::endl;
    return;
  }
  trialName = trialName.substr(0, trialName.find("_analog"));

  // Derive full paths
  fs::path numberDir = analogFile.parent_path().parent_path();
  // std::cout << numberDir << std::endl;
  fs::path imuDir = numberDir / "imu";
  fs::path mocapDir = numberDir / "mocap";

  fs::path orientationsFile = imuDir / (trialName + "_orientations.sto");
  fs::path accelerationsFile = imuDir / (trialName + "_accelerations.sto");
  fs::path grfFile = mocapDir / (trialName + "_grfs.sto");
  fs::path markerFile = mocapDir / (trialName + "_markers.trc");

  // Output paths

  // Helper lambda to swap original root for newRoot
  auto makeOutPath = [&](const fs::path &fullPath) {
    fs::path rel = fs::relative(fullPath, numberDir.parent_path());
    return newRoot / rel;
  };

  // Output paths with "extracted" replaced by "aligned"
  fs::path outOrientations = makeOutPath(orientationsFile);
  fs::path outAccelerations = makeOutPath(accelerationsFile);
  fs::path outAnalog = makeOutPath(analogFile);
  fs::path outGrf = makeOutPath(grfFile);
  fs::path outMarker = makeOutPath(markerFile);

  if (!fs::exists(orientationsFile) || !fs::exists(accelerationsFile)) {
    std::cerr << "Missing IMU files for trial: " << trialName << std::endl;
    return;
  }

  // Load analog and find trigger times
  OpenSim::TimeSeriesTable analog(analogFile.string());
  OpenSim::TimeSeriesTableQuaternion orientations(orientationsFile.string());

  const auto &times = analog.getIndependentColumn();
  const auto &times_orientations = orientations.getIndependentColumn();
  const auto &data = analog.getDependentColumn("trigger");

  std::optional<double> tStart;
  std::optional<double> tEnd;
  const double threshold = 0.5;
  const double earlyWindow = 5.0;
  const double minGap = 0.5; // reject noisy duplicate triggers

  std::vector<double> risingEdges;
  // Detect rising edges
  for (size_t i = 1; i < times.size(); ++i) {
    double prev = data(i - 1);
    double curr = data(i);

    if (prev <= threshold && curr > threshold) {
      double t = times[i];

      // Debounce
      if (!risingEdges.empty() && (t - risingEdges.back()) < minGap) {
        continue;
      }

      risingEdges.push_back(t);
    }
  }

  // Need to drop the first edge for this one because it wasn't accurate.
  if (trialName == "jogging" && numberDir.stem().string() == "08") {
    std::cout << "Special handling for 08 jogging trial!" << std::endl;
    risingEdges.erase(risingEdges.begin());
  }

  // Interpret edges
  // std::cout << "Trial: " << trialName << " " << "Number: " <<
  // numberDir.stem().string() << std::endl;
  if (risingEdges.size() >= 2) {
    tStart = risingEdges[0];
    tEnd = risingEdges[1];
  } else if (risingEdges.size() == 1) {
    double t = risingEdges[0];

    if (t < earlyWindow) {
      // Only start trigger present
      tStart = t;
      tEnd = times_orientations.back();
    } else {
      // Only end trigger present
      tEnd = t;

      // Estimate start relative to orientation duration
      double estimatedStart = tEnd.value() - times_orientations.back();
      // std::cout << "Calculated estimatedStart: " << estimatedStart << std::endl;

      if (estimatedStart >= 0.0) {
        tStart = estimatedStart;
      }
    }

  } else {
    throw std::invalid_argument(
        "Rising edges found: " + std::to_string(risingEdges.size()) + "\n");
  }

  // Validation
  if (!tStart || !tEnd || *tEnd <= *tStart) {
    throw std::invalid_argument(
        std::string("Invalid trigger signal in: ") +
        " start: " + (tStart ? std::to_string(*tStart) : "none") +
        " end: " + (tEnd ? std::to_string(*tEnd) : "none") + "\n");
  }

  const double rawStart = *tStart;
  const double rawEnd = *tEnd;

  // Create output directory if it doesn't exist
  fs::create_directories(outAnalog.parent_path());
  fs::create_directories(outOrientations.parent_path());

  // write IMU files
  OpenSim::TimeSeriesTableVec3 accelerations(accelerationsFile.string());
  OpenSim::STOFileAdapterQuaternion::write(orientations,
                                           outOrientations.string());
  OpenSim::STOFileAdapterVec3::write(accelerations, outAccelerations.string());

  // Trim markers (100 Hz) and get new cut time
  OpenSim::TimeSeriesTableVec3 marker_table(markerFile.string());
  const auto [markerStart, markerEnd] = trimAndWrite<OpenSim::TRCFileAdapter>(
      marker_table, outMarker, rawStart, rawEnd);

  const auto &imu_elapsed =
      times_orientations.back() - times_orientations.front();
  const auto &marker_elapsed = markerEnd - markerStart;
  std::cout << "File: " << markerFile.string()
            << " Raw elapsed: " << *tEnd - *tStart
            << " IMU elapsed: " << imu_elapsed
            << " Marker elapsed: " << marker_elapsed
            << " diff: " << marker_elapsed - imu_elapsed << std::endl;

  // Trim analog and grf (2400 Hz) signals to same time
  trimAndWrite<OpenSim::STOFileAdapter>(analog, outAnalog, markerStart,
                                        markerEnd);

  OpenSim::TimeSeriesTable grf_table(grfFile.string());
  trimAndWrite<OpenSim::STOFileAdapter>(grf_table, outGrf, markerStart,
                                        markerEnd);
}

void processDirectory(const fs::path &originalRoot, const fs::path &newRoot,
                      const unsigned &maxThreads =
                          std::max(1u, std::thread::hardware_concurrency())) {
  std::cout << "Processing with max threads: " << maxThreads << std::endl;
  std::counting_semaphore<> sem(maxThreads);
  std::vector<std::future<void>> futures;

  for (const auto &entry : fs::recursive_directory_iterator(
           originalRoot, fs::directory_options::skip_permission_denied)) {

    if (!entry.is_regular_file())
      continue;

    const auto path = entry.path();

    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    const auto name = path.filename().string();

    if (ext == ".sto" && name.find("_analog") != std::string::npos) {

      sem.acquire();
      futures.emplace_back(
          std::async(std::launch::async, [path, originalRoot, newRoot, &sem]() {
            try {
              processTrial(path, originalRoot, newRoot);
            } catch (const std::exception &e) {
              std::cerr << "Error processing " << path << ": " << e.what()
                        << std::endl;
            }
            sem.release();
          }));
    }
  }

  // Wait for all threads to complete
  for (auto &f : futures) {
    f.wait();
  }
}

int main(int argc, char *argv[]) {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <directory_path> <output_path>"
              << std::endl;
    return 1;
  }

  fs::path directoryPath = argv[1];
  if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
    std::cerr << "The provided path is not a valid directory." << std::endl;
    return 1;
  }

  fs::path outputPath = argv[2];
  if (!std::filesystem::exists(outputPath)) {
    // Create the directory
    if (std::filesystem::create_directories(outputPath)) {
      std::cout << "Directories created: " << outputPath << std::endl;
    } else {
      std::cout << "Failed to create directory: " << outputPath << std::endl;
    }
  }

  processDirectory(directoryPath, outputPath, MAX_THREADS);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Runtime = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[µs]" << std::endl;
  std::cout << "Results Saved to directory: " << outputPath << std::endl;
  std::cout << "Finished Running without Error!" << std::endl;
  return 0;
}
