import os
import fsspec_xrootd as xrdfs

def get_rootfiles(path, use_local=True, hostid='cmsdata.phys.cmu.edu'):
    """
    Unified function to retrieve .root files either locally or via XRootD.

    Args:
        path (str): 
            - If use_local=True, path is a local directory to search recursively.
            - If use_local=False, path is the XRootD directory to search.
        use_local (bool): If True, read local .root files; otherwise use XRootD logic.
        hostid (str): Host for XRootD (e.g. 'cmsdata.phys.cmu.edu').

    Returns:
        list of str: Full file paths or XRootD URLs for .root files.
    """
    if use_local:
        root_files = []
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                if f.endswith(".root"):
                    root_files.append(os.path.join(dirpath, f))
        return root_files
    else:
        # XRootD-based approach
        fs = xrdfs.XRootDFileSystem(hostid=hostid)
        return _get_files_recursive(
            fs,
            path,
            allowed=lambda f: f.endswith(".root"),
            prepend=f'root://{hostid}/'
        )

def _get_files_recursive(fs, rootpath, allowed=lambda f: True, prepend=''):
    pathlist = fs.ls(rootpath)
    result = []
    for p in pathlist:
        if p['type'] == 'directory':
            result += _get_files_recursive(fs, p['name'], allowed, prepend)
        elif p['type'] == 'file':
            if allowed(p['name']):
                result.append(prepend + p['name'])
        else:
            raise RuntimeError(f"Unexpected file type: {p['type']}")
    return result
