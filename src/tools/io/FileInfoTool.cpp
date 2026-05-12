#include "FileInfoTool.h"

#include <filesystem>

namespace fs = std::filesystem;

FileInfoTool::FileInfoTool(const std::string &base_dir) : base_directory(base_dir) {}

ToolDefinition FileInfoTool::get_definition() const {
    return {
        "file_info",
        "Get detailed information about a file (size, modification date, type)",
        {
            {"filepath", ToolArgType::String, "Path to the file (relative to base directory)", true}
        }
    };
}

ToolResult FileInfoTool::execute(const json &arguments) {
    ToolResult result;
    try {
        if (!arguments.contains("filepath") || !arguments["filepath"].is_string()) {
            result.error = "Missing or invalid 'filepath' parameter";
            return result;
        }
        std::string filepath = arguments["filepath"].get<std::string>();
        if (filepath.find("..") != std::string::npos) {
            result.error = "Invalid filepath: directory traversal not allowed";
            return result;
        }

        std::string full_path = base_directory + "/" + filepath;

        if (!fs::exists(full_path)) {
            result.error = "File does not exist: " + full_path;
            return result;
        }

        std::stringstream ss;
        ss << "File: " << filepath << "\n";
        ss << "Type: " << (fs::is_directory(full_path) ? "Directory" : "File") << "\n";

        if (!fs::is_directory(full_path)) {
            auto size = fs::file_size(full_path);
            ss << "Size: " << size << " bytes";
            if (size > 1024 * 1024) {
                ss << " (" << (double)(size / (1024.0 * 1024.0)) << " MB)";
            } else if (size > 1024) {
                ss << " (" << (double)(size / 1024.0) << " KB)";
            }
            ss << "\n";
        }

        auto ftime = fs::last_write_time(full_path);
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
        auto cftime = std::chrono::system_clock::to_time_t(sctp);

        ss << "Last modified: " << ctime(&cftime);

        auto perms = fs::status(full_path).permissions();
        ss << "Readable: " << ((perms & fs::perms::owner_read) != fs::perms::none ? "Yes" : "No") << "\n";
        ss << "Writeable: " << ((perms & fs::perms::owner_write) != fs::perms::none ? "Yes" : "No") << "\n";
        ss << "Executable: " << ((perms & fs::perms::owner_exec) != fs::perms::none ? "Yes" : "No") << "\n";

        result.result = ss.str();
        result.success = true;
        result.metadata["filepath"] = filepath;
        result.metadata["exists"] = true;
    }
    catch (const std::exception & e) {
        result.error = std::string("File info error: ") + e.what();
    }
    return result;
}