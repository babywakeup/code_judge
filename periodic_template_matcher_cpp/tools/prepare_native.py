"""Prepare a local reproducible OpenCV SDK; no system PATH/registry changes."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / '.deps'


def download(url, target):
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url}', flush=True)
    temporary = target.with_suffix(target.suffix + '.part')
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output)
    temporary.replace(target)


def run(args):
    print(' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=min(os.cpu_count() or 2, 6))
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--opencv-version', default='4.14.0',
                        help='Match the Python cv2 version for parity (default: 4.14.0)')
    args = parser.parse_args()
    if not all(part.isdigit() for part in args.opencv_version.split('.')):
        parser.error('opencv-version must contain only numeric version components')
    DEPS.mkdir(parents=True, exist_ok=True)
    download('https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp',
             DEPS / 'include/nlohmann/json.hpp')
    # NumPy v2.5.4 numpy/fft/pocketfft submodule, not the moving cpp branch.
    download('https://raw.githubusercontent.com/mreineck/pocketfft/'
             '33ae5dc94c9cdc7f1c78346504a85de87cadaa12/pocketfft_hdronly.h',
             DEPS / 'include/pocketfft/pocketfft_hdronly.h')
    archive = DEPS / f'opencv-{args.opencv_version}.zip'
    download(f'https://codeload.github.com/opencv/opencv/zip/refs/tags/{args.opencv_version}', archive)
    source = DEPS / f'opencv-{args.opencv_version}'
    if not source.exists():
        with zipfile.ZipFile(archive) as handle:
            # Python zipfile normalizes paths; additionally reject traversal entries.
            for name in handle.namelist():
                if '..' in Path(name).parts or Path(name).is_absolute():
                    raise ValueError('Unsafe archive path')
            handle.extractall(DEPS)
    if args.download_only:
        return
    cmake = Path(sys.executable).parent / ('cmake.exe' if os.name == 'nt' else 'cmake')
    if not cmake.exists():
        cmake = shutil.which('cmake')
    if not cmake:
        raise SystemExit('Install cmake and ninja in this Python environment first')
    extra = []
    if os.name == 'nt' and not shutil.which('cl'):
        zig = ROOT.parent / 'c++_speed_test/build/toolchain/ziglang/zig.exe'
        if not zig.exists():
            run([sys.executable, '-m', 'pip', 'install', 'ziglang==0.17.1', '--target', DEPS / 'toolchain'])
            zig = DEPS / 'toolchain/ziglang/zig.exe'
        tools = DEPS / 'tools'
        tools.mkdir(exist_ok=True)
        for name, command in [('cc', 'cc'), ('cxx', 'c++'), ('rc', 'rc'), ('ar', 'ar'), ('ranlib', 'ranlib')]:
            (tools / f'{name}.cmd').write_text(f'@"{zig}" {command} %*\n')
        (tools / 'archive.cmake').write_text(
            f'set(CMAKE_AR "{(tools / "ar.cmd").as_posix()}")\n'
            f'set(CMAKE_RANLIB "{(tools / "ranlib.cmd").as_posix()}")\n')
        extra = [f'-DCMAKE_C_COMPILER={(tools / "cc.cmd").as_posix()}',
                 f'-DCMAKE_CXX_COMPILER={(tools / "cxx.cmd").as_posix()}',
                 f'-DCMAKE_RC_COMPILER={(tools / "rc.cmd").as_posix()}',
                 f'-DCMAKE_AR={(tools / "ar.cmd").as_posix()}',
                 f'-DCMAKE_RANLIB={(tools / "ranlib.cmd").as_posix()}',
                 f'-DCMAKE_PROJECT_INCLUDE={(tools / "archive.cmake").as_posix()}']
    ninja = Path(sys.executable).parent / ('ninja.exe' if os.name == 'nt' else 'ninja')
    if ninja.exists():
        extra.append(f'-DCMAKE_MAKE_PROGRAM={ninja}')
    options = [
        '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_LIST=core,imgproc,photo,video',
        '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTS=OFF', '-DBUILD_PERF_TESTS=OFF',
        '-DBUILD_EXAMPLES=OFF', '-DBUILD_opencv_apps=OFF', '-DBUILD_opencv_python3=OFF',
        '-DBUILD_opencv_python2=OFF', '-DBUILD_JAVA=OFF', '-DWITH_IPP=OFF', '-DWITH_OPENCL=OFF',
        '-DWITH_ITT=OFF', '-DWITH_FFMPEG=OFF', '-DWITH_MSMF=OFF', '-DWITH_DSHOW=OFF',
        '-DWITH_PROTOBUF=OFF', '-DWITH_ADE=OFF', '-DBUILD_opencv_dnn=OFF',
        '-DWITH_JPEG=OFF', '-DWITH_PNG=OFF', '-DWITH_TIFF=OFF', '-DWITH_WEBP=OFF',
        '-DWITH_OPENEXR=OFF', '-DWITH_JASPER=OFF', '-DWITH_OPENJPEG=OFF', '-DWITH_AVIF=OFF',
        '-DOPENCV_ENABLE_NONFREE=OFF', '-DENABLE_PRECOMPILED_HEADERS=OFF',
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DOPENCV_SKIP_PYTHON_LOADER=ON',
        f'-DCMAKE_INSTALL_PREFIX={DEPS / "opencv-install"}',
    ]
    build = DEPS / f'opencv-build-{args.opencv_version}'
    run([cmake, '-S', source, '-B', build, '-G', 'Ninja', *options, *extra])
    run([cmake, '--build', build, '--target', 'install', '--parallel', args.jobs])


if __name__ == '__main__':
    main()
