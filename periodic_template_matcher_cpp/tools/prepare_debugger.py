"""Download portable CodeLLDB for verified local Windows native debugging."""
from prepare_native import DEPS, download
import zipfile


def main():
    archive=DEPS/'codelldb-win32-x64-1.12.3.vsix'
    download('https://github.com/vadimcn/codelldb/releases/download/v1.12.3/codelldb-win32-x64.vsix',archive)
    directory=DEPS/'codelldb'
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(directory)
    print(f'Debugger extracted to {directory}')


if __name__=='__main__':main()
