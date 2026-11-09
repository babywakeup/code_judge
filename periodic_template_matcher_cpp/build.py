"""Configure/build using a local OpenCV SDK or --opencv-dir from another SDK."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',choices=['Debug','Release','RelWithDebInfo'],default='Debug')
    parser.add_argument('--opencv-dir',type=Path)
    parser.add_argument('--jobs',type=int,default=min(os.cpu_count() or 2,6))
    parser.add_argument('--configure-only',action='store_true')
    args=parser.parse_args()
    scripts=Path(sys.executable).parent
    cmake=scripts/('cmake.exe' if os.name=='nt' else 'cmake')
    if not cmake.exists():cmake=shutil.which('cmake')
    if not cmake:raise SystemExit('Install cmake, ninja and pybind11 in the active Python environment')
    import pybind11
    directory=ROOT/'build'/args.config.lower()
    opencv=args.opencv_dir
    if opencv is None:
        found=list((ROOT/'.deps/opencv-install').rglob('OpenCVConfig.cmake'))
        if not found:raise SystemExit('OpenCV C++ SDK absent. Run tools/prepare_native.py or pass --opencv-dir')
        # The Windows install root is a dispatcher that can reject Clang/Zig.
        # Prefer the actual SDK config next to its libraries, not that dispatcher.
        opencv=max(found,key=lambda path:len(path.parts)).parent
    extra=[]
    tools=ROOT/'.deps/tools'
    if os.name=='nt' and (tools/'cxx.cmd').exists() and not shutil.which('cl'):
        for key,file in [('CMAKE_CXX_COMPILER','cxx.cmd'),('CMAKE_RC_COMPILER','rc.cmd'),('CMAKE_AR','ar.cmd'),('CMAKE_RANLIB','ranlib.cmd'),('CMAKE_PROJECT_INCLUDE','archive.cmake')]:
            extra.append(f'-D{key}={(tools/file).as_posix()}')
    ninja=scripts/('ninja.exe' if os.name=='nt' else 'ninja')
    if ninja.exists():extra.append(f'-DCMAKE_MAKE_PROGRAM={ninja.as_posix()}')
    command=[str(cmake),'-S',str(ROOT),'-B',str(directory),'-G','Ninja',
             f'-DCMAKE_BUILD_TYPE={args.config}',f'-DOpenCV_DIR={opencv.as_posix()}',
             f'-DPython_EXECUTABLE={Path(sys.executable).as_posix()}',
             f'-Dpybind11_DIR={pybind11.get_cmake_dir()}',*extra]
    subprocess.run(command,check=True)
    if not args.configure_only:
        subprocess.run([str(cmake),'--build',str(directory),'--parallel',str(args.jobs)],check=True)
        extension=next((directory/'python').glob('_native*.pyd' if os.name=='nt' else '_native*.so'),None)
        if extension is None:
            raise SystemExit('Build succeeded but the Python extension was not produced')
        destination=ROOT/extension.name
        try:
            shutil.copy2(extension,destination)
            symbols=extension.with_suffix('.pdb')
            if symbols.exists():shutil.copy2(symbols,ROOT/symbols.name)
        except PermissionError as error:
            raise SystemExit('Close Python processes using _native, then run build.py again to publish it') from error
        report=dict(configuration=args.config,python=sys.version,opencv_dir=str(opencv),
                    artifact=str(extension.relative_to(ROOT)),
                    sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
        (ROOT/'build/active-native.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(f'Published {args.config}: {destination}')


if __name__=='__main__':main()
