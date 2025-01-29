import fsspec_xrootd as xrdfs
import os

def get_rootfiles(path):
    root_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(".root"):
                root_files.append(os.path.join(dirpath, f))
    return root_files

def get_files_recursive(fs, rootpath, allowed = lambda f : True, prepend = ''):
    pathlist = fs.ls(rootpath)
    result = []
    for path in pathlist:
        if path['type'] == 'directory':
            result += get_files_recursive(fs, path['name'], allowed, prepend)
        elif path['type'] == 'file':
            if allowed(path['name']):
                result.append(prepend + path['name'])
        else:
            raise RuntimeError("Unexpected file type: {}".format(path['type']))
    return result
