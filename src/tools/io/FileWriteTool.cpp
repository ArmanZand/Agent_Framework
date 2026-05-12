#include "FileWriteTool.h"
#include <fstream>

FileWriteTool::FileWriteTool(const std::string &base_dir) : base_directory(base_dir) {}

ToolDefinition FileWriteTool::get_definition() const {
    return {
        "write_file",
        "Write the contents of a text file",
        {{"filepath", ToolArgType::String, "Path to the file to write", true},
        {"content", ToolArgType::String, "Content to write to the file", true},
        }
    };
}

ToolResult FileWriteTool::execute(const json &arguments) {
    ToolResult result;

    try {
        if (!arguments.contains("filepath") || !arguments["filepath"].is_string()) {
            result.error = "Missing or invalid 'filepath' parameter";
            return result;
        }

        if (!arguments.contains("content") || !arguments["content"].is_string()) {
            result.error = "Missing or invalid 'content' parameter";
            return result;
        }

        std::string filepath = arguments["filepath"].get<std::string>();
        std::string content = arguments["content"].get<std::string>();

        if (filepath.find("..") != std::string::npos) {
            result.error = "Invalid filepath: directory traversal not allowed";
            return result;
        }

        std::string full_path = base_directory + "/" + filepath;

        std::ofstream file(full_path);
        if (!file.is_open()) {
            result.error = "Failed to open file for writing: " + filepath;
            return result;
        }
        file << content;
        file.close();

        result.result = "Successfully wrote" + std::to_string(content.length()) + " bytes to " + filepath;
        result.success = true;
        result.metadata["filepath"] = filepath;
        result.metadata["size_bytes"] = std::to_string(content.length());

    }
    catch (const std::exception & e) {
        result.error = std::string("File write error: ") + e.what();
    }
    return result;
}