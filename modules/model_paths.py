import os


def find_file_in_folder_list(name, folders, recursive=False):
    if not isinstance(folders, list):
        folders = [folders]

    for folder in folders:
        filename = os.path.abspath(os.path.realpath(os.path.join(folder, name)))
        if os.path.isfile(filename):
            return filename

    if recursive:
        target_name = os.path.basename(name).casefold()
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            for root, directories, filenames in os.walk(folder):
                directories.sort(key=str.casefold)
                for filename in sorted(filenames, key=str.casefold):
                    if filename.casefold() == target_name:
                        return os.path.abspath(os.path.realpath(os.path.join(root, filename)))

    return None


def get_file_name_from_folder_list(filename, folders):
    if not isinstance(folders, list):
        folders = [folders]

    filename = os.path.abspath(os.path.realpath(filename))
    for folder in folders:
        folder = os.path.abspath(os.path.realpath(folder))
        try:
            if os.path.commonpath([filename, folder]) == folder:
                return os.path.relpath(filename, folder)
        except ValueError:
            continue

    return os.path.basename(filename)
