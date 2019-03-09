import os
import glob


def exist_directory(directory_path):
    return os.path.isdir(directory_path)


def exist_file(file_path):
    return os.path.isfile(file_path)


def construct_path(directory_path, directory_content):
    return os.path.join(directory_path, directory_content)


def get_directory_contents(directory_path, pattern='*'):
    if exist_directory(directory_path):
        cwd = os.getcwd()
        # change current wording directory
        os.chdir(directory_path)
        # file list
        file_names = glob.glob(pattern)
        # change the cwd back to normal
        os.chdir(cwd)
        return file_names
    return []


def remove_pattern(orig_str, pattern):
    return orig_str.replace(pattern, '')


def get_subdirectory_names(directory_path):
    return get_directory_contents(directory_path, '*//')


def get_subdirectory_nms(directory_path):
    return set([remove_pattern(subdirectory, '//') for subdirectory in get_subdirectory_names(directory_path)])


def get_file_names(directory_path, extension='.*'):
    return get_directory_contents(directory_path, f'*{extension}')


def get_file_nms(directory_path, extension='.*'):
    return set([remove_pattern(file_name, extension) for file_name in get_file_names(directory_path, extension)])
