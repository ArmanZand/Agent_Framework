#include "ExploreDirectoryTool.h"

#include <filesystem>

namespace fs = std::filesystem;
ExploreDirectoryTool::ExploreDirectoryTool(const std::string &base_dir) : base_directory(base_dir) {}

ToolDefinition ExploreDirectoryTool::get_definition() const {
    return {
    "explore_directory",
    "List all files and subdirectories in a directory, can also recursively scan.",
    {
        {"path", ToolArgType::String, "Directory path to list (relative to base directory). Use '.' for current directory.", false, "."},
        {"recursive", ToolArgType::Boolean, "Whether to list subdirectories recursively", false, false}
    }};
}

ToolResult ExploreDirectoryTool::execute(const json &arguments) {
    ToolResult result;
    try {
        std::string path = ".";
        if (arguments.contains("path") && arguments["path"].is_string()) {
            path = arguments["path"].get<std::string>();
        }

        bool recursive = false;
        if (arguments.contains("recursive") && arguments["recursive"].is_boolean()) {
            recursive = arguments["recursive"].get<bool>();
        }

        if (path.find("..") != std::string::npos) {
            result.error = "Invalid path: directory traversal not allowed";
            return result;
        }

        std::string full_path = base_directory;
        if (path == "." || path.empty()) {
            full_path = base_directory;
        } else {
            full_path = base_directory + "/" + path;
        }

        if (!fs::exists(full_path)) {
            result.error = "Directory does not exist: " + full_path;
            return result;
        }
        if (!fs::is_directory(full_path)) {
            result.error = "Path is not a directory: " + full_path;
            return result;
        }

        std::stringstream ss;
        ss << "Directory Tree: " << (path == "." ? "ROOT FOLDER" : path) << "\n";
        ss << std::string(60, '=') << "\n";

        int file_count = 0;
        int dir_count = 0;

        if (recursive) {
            std::map<std::string, std::vector<std::string>> dir_contents;
            std::map<std::string, std::vector<std::pair<std::string, size_t>>> file_contents;
            for (const auto & entry : fs::recursive_directory_iterator(full_path)) {
                std::string relative = fs::relative(entry.path(), full_path).string();

                fs::path rel_path(relative);
                std::string parent = rel_path.parent_path().string();
                if (parent.empty()) parent = ".";

                std::string name = rel_path.filename().string();

                if (entry.is_directory()) {
                    dir_contents[parent].push_back(name);
                    dir_count++;
                } else {
                    auto size = fs::file_size(entry.path());
                    file_contents[parent].emplace_back(name, size);
                    file_count++;
                }
            }

            auto print_level = [&](const std::string & level_path, int depth, auto & print_ref) -> void {
                const std::string indent = std::string(depth * 2, ' ');
                size_t last_slash = level_path.rfind('/');
                const std::string display_name = (last_slash == std::string::npos) ? level_path : level_path.substr(last_slash + 1);
                const std::string display_path = (level_path == ".") ? "." : display_name;
                ss << indent << "📁 " << display_path << "/\n";

                if (file_contents.find(level_path) != file_contents.end()) {
                    for (const auto & [filename, size] : file_contents[level_path]) {
                        ss << indent << " ├─ 📄 " << filename << " (" << size << " bytes)";
                        std::string full_file_path = (level_path == ".") ? filename : level_path + "/" + filename;
                        ss << " [path: " << full_file_path << "]\n";
                    }
                }

                if (dir_contents.find(level_path) != dir_contents.end()) {
                    for (const auto & subdir : dir_contents[level_path]) {
                        std::string subdir_path = (level_path == ".") ? subdir : level_path  + "/" + subdir;
                        print_ref(subdir_path, depth + 1, print_ref);
                    }
                }

            };

            print_level(".", 0, print_level);

        }
        else {
            std::vector<std::string> directories;
            std::vector<std::pair<std::string, size_t>> files;

            for (const auto & entry : fs::directory_iterator(full_path)) {
                std::string name = entry.path().filename().string();

                if (entry.is_directory()) {
                    directories.push_back(name);
                    dir_count++;
                }
                else {
                    auto size = fs::file_size(entry.path());
                    files.emplace_back(name, size);
                    file_count++;
                }
            }

            if (!files.empty()) {
                ss << "📄 Files in " << path << ":\n";
                for (const auto & [name, size] : files) {
                    std::string file_path = (path == ".") ? name : path + "/" + name;
                    ss << " ├─ " << name << " (" << size << " bytes) [use path: '" << file_path << "' to read]\n";
                }
                ss << "\n";
            }

            if (!directories.empty()) {
                ss << "📁 Subdirectories in " << path << ":\n";
                for (const auto & dir : directories) {
                    std::string file_path = (path == ".") ? dir : path + "/" + dir;
                    ss << " ├─ " << dir << "/ [use path: '" << file_path << "' to list contents]\n";
                }
            }

            if (directories.empty() && files.empty()) {
                ss << "(empty directory)\n\n";
            }
        }

        ss << std::string(60, '=') << "\n";
        ss << "\nTotal: " << dir_count << " directories, " << file_count << " files\n";

        result.result = ss.str();
        result.success = true;
        result.metadata["path"] = path;
        result.metadata["file_count"] = std::to_string(file_count);
        result.metadata["dir_count"] = std::to_string(dir_count);
    }
    catch (const std::exception & e) {
        result.error = std::string("Directory listing error: ") + e.what();
    }
    return result;
}