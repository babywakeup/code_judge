"""Build the small C++ shared library without Python binding dependencies."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / ("matcher.dll" if sys.platform == "win32" else
                  "libmatcher.dylib" if sys.platform == "darwin" else "libmatcher.so")


def build():
    source = str(ROOT / "matcher.cpp")
    local_zig = ROOT / "build" / "toolchain" / "ziglang" / ("zig.exe" if sys.platform == "win32" else "zig")
    compiler = next((shutil.which(name) for name in ("g++", "clang++")
                     if shutil.which(name)), None)
    if compiler or local_zig.exists():
        command = [compiler] if compiler else [str(local_zig), "c++"]
        flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
        if sys.platform != "win32":
            flags += ["-fPIC"]
        subprocess.run([*command, "-O3", "-std=c++17", *flags, source,
                        "-o", str(LIBRARY)], check=True, cwd=ROOT)
    elif sys.platform == "win32":
        # setuptools discovers installed MSVC even outside a Developer Prompt.
        try:
            import setuptools  # activates its bundled distutils on Python 3.12+
            from distutils.errors import DistutilsError
            from distutils import ccompiler
            try:
                from distutils.compilers.errors import Error as CompilerError
            except ImportError:
                CompilerError = DistutilsError
            compiler = ccompiler.new_compiler(compiler="msvc")
            compiler.initialize()
            objects = compiler.compile([source], output_dir=str(ROOT / "build"),
                                       extra_postargs=["/O2", "/std:c++17", "/EHsc"])
            compiler.link_shared_object(objects, str(LIBRARY))
        except ImportError as exc:
            raise SystemExit("Install setuptools, or put g++/clang++ on PATH.") from exc
        except (OSError, RuntimeError, DistutilsError, CompilerError) as exc:
            raise SystemExit("Build failed. Install MSVC Build Tools (Desktop development "
                             "with C++) and setuptools, or put g++/clang++ on PATH.\n"
                             + str(exc)) from exc
    else:
        raise SystemExit("No C++ compiler found. Install g++ or clang++ first.")
    print(f"Built: {LIBRARY}")


if __name__ == "__main__":
    build()
