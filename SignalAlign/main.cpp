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
#include <Common/TimeSeriesTable.h>
#include <OpenSim/Common/C3DFileAdapter.h>
#include <OpenSim/Common/STOFileAdapter.h>
#include <OpenSim/Common/TRCFileAdapter.h>

#include <SimTKcommon/SmallMatrix.h>
#include <SimTKcommon/internal/Quaternion.h>
#include <chrono> // for std::chrono functions
#include <filesystem>
#include <future> // For std::async, std::future
#include <iostream>
#include <string>

namespace fs = std::filesystem;

// Trim a TimeSeriesTable to a time window and write to new location
std::pair<double, double> trimAndWrite(const fs::path &inFile,
                                       const fs::path &outFile, double tStart,
                                       double tEnd) {
  // Create a file adapter instance
  OpenSim::TimeSeriesTable table(inFile.string());
  table.trim(tStart, tEnd);
  OpenSim::STOFileAdapter::write(table, outFile.string());
  std::cout << "Trimmed and saved: " << outFile << std::endl;
  // Get new time range
  const auto &newTimes = table.getIndependentColumn();
  if (newTimes.empty()) {
    std::cerr << "Warning: Trimmed table is empty." << std::endl;
    return {-1.0, -1.0};
  }

  double newStart = newTimes.front();
  double newEnd = newTimes.back();
  return {newStart, newEnd};
}

std::pair<double, double> trimAndWriteVec3(const fs::path &inFile,
                                           const fs::path &outFile,
                                           double tStart, double tEnd) {
  // Create a file adapter instance
  OpenSim::TimeSeriesTableVec3 table(inFile.string());
  table.trim(tStart, tEnd);
  OpenSim::STOFileAdapterVec3::write(table, outFile.string());
  std::cout << "Trimmed and saved: " << outFile << std::endl;
  // Get new time range
  const auto &newTimes = table.getIndependentColumn();
  if (newTimes.empty()) {
    std::cerr << "Warning: Trimmed table is empty." << std::endl;
    return {-1.0, -1.0};
  }

  double newStart = newTimes.front();
  double newEnd = newTimes.back();
  return {newStart, newEnd};
}

std::pair<double, double> trimAndWriteTrc(const fs::path &inFile,
                                           const fs::path &outFile,
                                           double tStart, double tEnd) {
  // Create a file adapter instance
  OpenSim::TimeSeriesTableVec3 table(inFile.string());
  table.trim(tStart, tEnd);
  OpenSim::TRCFileAdapter::write(table, outFile.string());
  std::cout << "Trimmed and saved: " << outFile << std::endl;
  // Get new time range
  const auto &newTimes = table.getIndependentColumn();
  if (newTimes.empty()) {
    std::cerr << "Warning: Trimmed table is empty." << std::endl;
    return {-1.0, -1.0};
  }

  double newStart = newTimes.front();
  double newEnd = newTimes.back();
  return {newStart, newEnd};
}

std::pair<double, double> trimAndWriteQuaternion(const fs::path &inFile,
                                                 const fs::path &outFile,
                                                 double tStart, double tEnd) {
  // Create a file adapter instance
  OpenSim::TimeSeriesTableQuaternion table(inFile.string());
  table.trim(tStart, tEnd);
  OpenSim::STOFileAdapterQuaternion::write(table, outFile.string());
  std::cout << "Trimmed and saved: " << outFile << std::endl;
  // Get new time range
  const auto &newTimes = table.getIndependentColumn();
  if (newTimes.empty()) {
    std::cerr << "Warning: Trimmed table is empty." << std::endl;
    return {-1.0, -1.0};
  }

  double newStart = newTimes.front();
  double newEnd = newTimes.back();
  return {newStart, newEnd};
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

  double tStart = 0.0;
  double tEnd = -1.0;
  if (!times.empty() && !times_orientations.empty()) {
    tEnd = (times.size() <= times_orientations.size())
               ? times.back()
               : times_orientations.back();
  }
  double threshold = 1;
  for (size_t i = 1; i < times.size(); ++i) {
    double timestamp = times.at(i);
    double prev = data(i - 1);
    double curr = data(i);
    if (prev <= threshold && curr > threshold) {
      if (tStart == 0 && timestamp < 10.0) {
        tStart = times[i];
      } else {
        tEnd = times[i];
        break;
      }
    }
  }

  if (tStart < 0 || tEnd < 0 || tEnd <= tStart) {
    std::cerr << "Invalid trigger signal in: " << analogFile
              << " start: " << tStart << " end: " << tEnd << std::endl;
    return;
  }

  std::cout << "Trim window for '" << trialName << "': " << tStart << "s to "
            << tEnd << "s\n";

  // Trim and write IMU files

  // Create output directory if it doesn't exist
  fs::create_directories(outAnalog.parent_path());
  fs::create_directories(outOrientations.parent_path());

  trimAndWriteQuaternion(orientationsFile, outOrientations, tStart, tEnd);
  auto [newStart, newEnd] =
      trimAndWriteVec3(accelerationsFile, outAccelerations, tStart, tEnd);

  trimAndWrite(analogFile, outAnalog, newStart, newEnd);
  trimAndWrite(grfFile, outGrf, newStart, newEnd);
  trimAndWriteTrc(markerFile, outMarker, newStart, newEnd);
}

void processDirectory(const fs::path &originalRoot, const fs::path &newRoot) {
  std::vector<std::future<void>> futures; // Store async tasks

  for (const auto &entry : fs::directory_iterator(originalRoot)) {
    if (entry.is_directory()) {
      // Recursively process subdirectory
      processDirectory(entry.path(), newRoot);
    } else if (entry.is_regular_file()) {
      const auto file = entry.path();

      if (file.extension() == ".sto" &&
          file.filename().string().find("_analog") != std::string::npos) {

        // Launch each trial in parallel using std::async
        futures.emplace_back(
            std::async(std::launch::async, [file, originalRoot, newRoot]() {
              try {
                processTrial(file, originalRoot, newRoot);
              } catch (const std::exception &e) {
                std::cerr << "Error processing " << file << ": " << e.what()
                          << std::endl;
              }
            }));
      }
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

  processDirectory(directoryPath, outputPath);
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
