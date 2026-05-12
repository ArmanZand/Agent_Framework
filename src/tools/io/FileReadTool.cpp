#include "FileReadTool.h"

#include <fstream>
#include <sstream>

FileReadTool::FileReadTool(const std::string &base_dir) : base_directory(base_dir) {}

ToolDefinition FileReadTool::get_definition() const {
    return {
        "read_file",
        "Read the contents of a text file",
        {
                {"filepath", ToolArgType::String, "Path to the file to read", true}
        }
    };
}

ToolResult FileReadTool::execute(const json &arguments) {
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

        std::ifstream file(full_path);
        if (!file.is_open()) {
            result.error = "Failed to open file: " + filepath;
            return result;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();

        result.result = buffer.str();
        result.success = true;
        result.metadata["filepath"] = filepath;
        result.metadata["size_bytes"] = std::to_string(result.result.length());
    }
    catch (const std::exception & e) {
        result.error = std::string("File read error: ") + e.what();

    }
    return result;
}