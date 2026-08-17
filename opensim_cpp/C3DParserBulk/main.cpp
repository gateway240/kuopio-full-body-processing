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

#include <chrono> // for std::chrono functions
#include <filesystem>
#include <iostream>
#include <regex> // for regex parsing
#include <string>

// Map from last ID number string to flipped label (Muscle_Side)
// Snap Lead Sensor 57176.Avanti Sensor 55485.EMG 1
std::map<std::string, std::string> idToLabel = {
    {"56697", "VL_Right"}, {"57644", "VL_Left"},
    {"57648", "RF_Right"}, {"56691", "RF_Left"},
    {"56748", "VM_Right"}, {"55485", "VM_Left"},
    {"85724", "BF_Right"}, {"57583", "BF_Left"},
    {"58147", "ST_Right"}, {"55128", "ST_Left"},
    {"54693", "LD_Right"}, {"55176", "LD_Left"},
    {"57176", "GM_Right"}, {"61514", "GM_Left"},
    {"62782", "Right"}, // Q1 group (TT, DM, TA, TD)
    {"62780", "Left"}   // Q2 group (TT, DM, TA, TD)
};

namespace fs = std::filesystem;

void processC3DFile(const fs::path &filename, const fs::path &resultPath) {
  std::cout << "---Starting Processing: " << filename << std::endl;
  try {
    OpenSim::C3DFileAdapter c3dFileAdapter{};
    c3dFileAdapter.setLocationForForceExpression(
        OpenSim::C3DFileAdapter::ForceLocation::CenterOfPressure);
    auto tables = c3dFileAdapter.read(filename);

    std::shared_ptr<OpenSim::TimeSeriesTableVec3> marker_table =
        c3dFileAdapter.getMarkersTable(tables);
    std::shared_ptr<OpenSim::TimeSeriesTableVec3> force_table =
        c3dFileAdapter.getForcesTable(tables);
    std::shared_ptr<OpenSim::TimeSeriesTable> analog_table =
        c3dFileAdapter.getAnalogDataTable(tables);
    // Additional mapping to append muscle names for Q1 and Q2 group sensors
    // We'll detect the original column name and add muscle-specific suffixes

    // Regex to extract the ID number from the analog column name
    std::regex idRegex(R"((\d{5}))"); // matches exactly 5 digits

    // Change PXXX:LFHD => LFHD
    if (marker_table) {
      auto colNames = marker_table->getColumnLabels();
      for (size_t i = 0; i < colNames.size(); ++i) {
        std::string colName = colNames[i];
        // Find the position of the colon
        size_t pos = colName.find(':');
        if (pos != std::string::npos) {
          // Keep only the part after the colon
          std::string newLabel = colName.substr(pos + 1);
          marker_table->setColumnLabel(i, newLabel);
        }
      }
    }

    if (analog_table) {
      auto colNames = analog_table->getColumnLabels();

      for (size_t i = 0; i < colNames.size(); ++i) {
        std::string colName = colNames[i];
        if (colName == "Electric Potential.StartStop") {
          // std::cout << colName << std::endl;
          analog_table->setColumnLabel(i, "trigger");
          continue; // skip further processing for this column
        }
          std::vector<std::string> matches;

          for (std::sregex_iterator it(colName.begin(), colName.end(), idRegex), end;
              it != end; ++it) {
              matches.push_back((*it).str(1)); // capture group 1
          }
          std::string idNum;
          if (matches.empty()) {
              // If ID not found in map, keep original column name
          } else if (matches.size() > 1) {
            std::cout << "Multiple matches found in colName: " << colName << std::endl;
            idNum = matches[0];
          } else {
            // Only one match, use it
            idNum = matches[0];
            auto it = idToLabel.find(idNum);
            std::string newLabel = it->second;
            // std::cout << newLabel << std::endl;

            // For the special Q1/Q2 groups with multiple muscles, append muscle
            // suffix
            if (idNum == "62782" || idNum == "62780") {
              // Check which muscle from the original label (contains EMG
              // channel number 1..4)
              std::string prefix;
              if (colName.find(".EMG 1") != std::string::npos) {
                prefix = "TT_";
              } else if (colName.find(".EMG 2") != std::string::npos) {
                prefix = "DM_";
              } else if (colName.find(".EMG 3") != std::string::npos) {
                prefix = "TA_";
              } else if (colName.find(".EMG 4") != std::string::npos) {
                prefix = "TD_";
              }
              newLabel = prefix + newLabel;
            }
            analog_table->setColumnLabel(i, newLabel);
        }
      }
    }

    // Get the last two parent directories
    std::filesystem::path firstParent = filename.parent_path(); // First parent

    std::filesystem::path secondParent =
        firstParent.parent_path(); // Second parent

    std::filesystem::path baseDir =
        resultPath / secondParent.filename() / firstParent.filename() / "";

    // Create directories if they don't exist
    try {
      if (std::filesystem::create_directories(baseDir)) {
        std::cout << "Directories created: " << baseDir << std::endl;
      }
    } catch (const std::filesystem::filesystem_error &e) {
      std::cerr << "Error creating directories: " << e.what() << std::endl;
    }

    const std::string marker_file =
        baseDir.string() + filename.stem().string() + "_markers.trc";
    const std::string forces_file =
        baseDir.string() + filename.stem().string() + "_grfs.sto";
    const std::string analogs_file =
        baseDir.string() + filename.stem().string() + "_analog.sto";

    // Write marker locations
    marker_table->updTableMetaData().setValueForKey("Units", std::string{"mm"});
    OpenSim::TRCFileAdapter trc_adapter{};
    trc_adapter.write(*marker_table, marker_file);
    std::cout << "\tWrote '" << marker_file << std::endl;

    analog_table->addTableMetaData("nRows",
                                   std::to_string(analog_table->getNumRows()));
    // Dependant columns + time
    analog_table->addTableMetaData(
        "nColumns", std::to_string(analog_table->getNumColumns() + 1));

    // These are not populated by our C3D files.
    force_table->removeTableMetaDataKey("Corners");
    force_table->removeTableMetaDataKey("CalibrationMatrices");
    force_table->removeTableMetaDataKey("Origins");
    force_table->removeTableMetaDataKey("Types");
    force_table->removeTableMetaDataKey("events");
    force_table->addTableMetaData("nRows",
                                  std::to_string(force_table->getNumRows()));
    // Dependant columns (vec3 will be flattened) * 3 + time
    force_table->addTableMetaData(
        "nColumns", std::to_string(force_table->getNumColumns() * 3 + 1));

    // Write forces and analog
    OpenSim::STOFileAdapter sto_adapter{};
    sto_adapter.write((force_table->flatten()), forces_file);
    std::cout << "\tWrote'" << forces_file << std::endl;
    sto_adapter.write(*analog_table, analogs_file);
    std::cout << "\tWrote'" << analogs_file << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error processing C3D file '" << filename << "': " << e.what()
              << std::endl;
  }
  std::cout << "---Ending Processing: " << filename << std::endl;
}

void processDirectory(const fs::path &dirPath, const fs::path &resultPath) {
  std::vector<std::thread> threads;
  // Iterate through the directory
  for (const auto &entry : fs::directory_iterator(dirPath)) {
    if (entry.is_directory()) {
      // Recursively process subdirectory
      processDirectory(entry.path(), resultPath);
    } else if (entry.is_regular_file()) {
      // Check if the file has a .c3d extension
      if (entry.path().extension() == ".c3d") {
        std::string fullPathStr = entry.path().string();
        // Create a corresponding text file
        fs::path textFilePath = entry.path();
        threads.emplace_back(processC3DFile, textFilePath, resultPath);
      }
    }
  }
  // Join all threads
  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
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
