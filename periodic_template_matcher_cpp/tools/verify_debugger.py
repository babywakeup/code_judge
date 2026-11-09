"""Verify a resolved AND hit source breakpoint using portable LLDB CLI or DAP TCP."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / ".deps/codelldb/extension"
MARKER = "PTM_BREAKPOINT_EVIDENCE="


def clean_environment() -> dict[str, str]:
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "CODELLDB_STARTUP"):
        env.pop(name, None)
    return env


def same_source(actual: str, expected: Path, source_map: list) -> bool:
    # SBLineEntry may retain the compile-time path even with target.source-map.
    # Compare the explicit longest-prefix remap, not just the displayed path.
    normalized = actual.replace("\\", "/")
    for old, new in sorted(source_map, key=lambda pair: len(pair[0]), reverse=True):
        prefix = old.replace("\\", "/").rstrip("/")
        if normalized.lower() == prefix.lower() or normalized.lower().startswith(prefix.lower() + "/"):
            normalized = new.replace("\\", "/").rstrip("/") + normalized[len(prefix):]
            break
    return os.path.normcase(os.path.abspath(normalized)) == os.path.normcase(str(expected.resolve()))


def verify_cli(args, line: int) -> dict:
    cli = args.extension / "lldb/bin/lldb.exe"
    if not cli.is_file():
        raise RuntimeError(f"LLDB CLI absent: {cli}")
    command = [str(cli), "--batch", "-o", "settings set auto-confirm true"]
    for old, new in args.source_map:
        command += ["-o", f'settings append target.source-map {json.dumps(old)} {json.dumps(new)}']
    command += ["-o", f'breakpoint set --file {json.dumps(args.source.name)} --line {line}',
                "-o", "run", "-o", "thread backtrace", "-o", "frame variable argc inputs"]
    # Query debugger state, not a regex on the echoed breakpoint command.
    expression = (
        "script import lldb,json; p=lldb.debugger.GetSelectedTarget().GetProcess(); "
        "t=p.GetSelectedThread(); f=t.GetFrameAtIndex(0); e=f.GetLineEntry(); "
        "print(" + repr(MARKER) + "+json.dumps(dict("
        "reason=int(t.GetStopReason()), breakpoint_reason=int(lldb.eStopReasonBreakpoint), "
        "source=e.GetFileSpec().fullpath, line=e.GetLine(), function=f.GetFunctionName(), "
        "argc=f.FindVariable('argc').GetValue(), inputs_valid=f.FindVariable('inputs').IsValid())))"
    )
    command += ["-o", expression, "-o", "continue", "-o",
                "script p=lldb.debugger.GetSelectedTarget().GetProcess(); "
                "print('PTM_EXIT_EVIDENCE='+json.dumps(dict(state=int(p.GetState()), "
                "exited_state=int(lldb.eStateExited), exit_code=p.GetExitStatus())))",
                "--", str(args.executable)]
    if args.fixture:
        command.append(str(args.fixture))
    completed = subprocess.run(command, cwd=ROOT.parent, env=clean_environment(),
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=args.timeout)
    transcript = completed.stdout + completed.stderr
    if args.log:
        args.log.write_text(transcript, encoding="utf-8")
    evidence = None
    exit_evidence = None
    for row in transcript.splitlines():
        if row.startswith(MARKER):
            evidence = json.loads(row[len(MARKER):])
        if row.startswith("PTM_EXIT_EVIDENCE="):
            exit_evidence = json.loads(row.split("=", 1)[1])
    if (completed.returncode != 0 or not evidence
            or evidence["reason"] != evidence["breakpoint_reason"]
            or not same_source(evidence["source"] or "", args.source, args.source_map)
            or evidence["line"] != line):
        raise RuntimeError("CLI did not prove the requested source breakpoint hit.\n" + transcript[-12000:])
    if (not exit_evidence or exit_evidence["state"] != exit_evidence["exited_state"]
            or exit_evidence["exit_code"] != 0):
        raise RuntimeError("Breakpoint hit, but debuggee did not exit successfully.\n" + transcript[-12000:])
    return dict(backend="lldb-cli", debugger=str(cli), executable=str(args.executable),
                fixture=str(args.fixture) if args.fixture else None,
                breakpoint=evidence, debuggee_exit=exit_evidence)


class DAP:
    def __init__(self, connection: socket.socket, timeout: float):
        self.connection = connection
        self.deadline = time.monotonic() + timeout
        self.buffer = b""
        self.sequence = 0
        self.pending = []
        self.transcript = []

    def send(self, command: str, arguments: dict) -> int:
        self.sequence += 1
        message = dict(seq=self.sequence, type="request", command=command, arguments=arguments)
        body = json.dumps(message).encode("utf-8")
        self.connection.sendall(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        self.transcript.append(message)
        return self.sequence

    def receive(self) -> dict:
        while True:
            header_end = self.buffer.find(b"\r\n\r\n")
            if header_end >= 0:
                headers = self.buffer[:header_end].decode("ascii")
                length = next(int(row.split(":", 1)[1]) for row in headers.split("\r\n")
                              if row.lower().startswith("content-length:"))
                end = header_end + 4 + length
                if len(self.buffer) >= end:
                    message = json.loads(self.buffer[header_end + 4:end])
                    self.buffer = self.buffer[end:]
                    self.transcript.append(message)
                    return message
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DAP verification deadline exceeded")
            self.connection.settimeout(remaining)
            chunk = self.connection.recv(65536)
            if not chunk:
                raise RuntimeError("CodeLLDB closed the DAP TCP connection")
            self.buffer += chunk

    def wait(self, predicate) -> dict:
        for index, message in enumerate(self.pending):
            if predicate(message):
                return self.pending.pop(index)
        while True:
            message = self.receive()
            if message.get("type") == "request":
                raise RuntimeError(f"Unexpected reverse request: {message}")
            if predicate(message):
                return message
            self.pending.append(message)

    def response(self, sequence: int) -> dict:
        message = self.wait(lambda m: m.get("type") == "response" and m.get("request_seq") == sequence)
        if not message.get("success"):
            raise RuntimeError(f"DAP request failed: {message}")
        return message.get("body", {})

    def request(self, command: str, arguments: dict) -> dict:
        return self.response(self.send(command, arguments))

    def event(self, name: str) -> dict:
        return self.wait(lambda m: m.get("type") == "event" and m.get("event") == name).get("body", {})


def verify_dap(args, line: int) -> dict:
    adapter = args.extension / "adapter/codelldb.exe"
    library = args.extension / "lldb/bin/liblldb.dll"
    if not adapter.is_file() or not library.is_file():
        raise RuntimeError(f"CodeLLDB adapter or liblldb.dll absent under {args.extension}")
    dap = None
    # Reverse TCP connection removes port-selection races and startup polling.
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(args.timeout)
        process = subprocess.Popen([str(adapter), "--liblldb", str(library), "--connect",
                                    str(server.getsockname()[1])], env=clean_environment(),
                                   cwd=ROOT.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            connection, _ = server.accept()
            with connection:
                dap = DAP(connection, args.timeout)
                dap.request("initialize", dict(adapterID="codelldb", clientID="ptm-verify",
                            linesStartAt1=True, columnsStartAt1=True, pathFormat="path",
                            supportsRunInTerminalRequest=False))
                launch = dap.send("launch", dict(program=str(args.executable),
                                  args=[str(args.fixture)] if args.fixture else [],
                                  cwd=str(ROOT.parent), stopOnEntry=False, terminal="console",
                                  sourceMap=dict(args.source_map)))
                dap.event("initialized")
                breakpoints = dap.request("setBreakpoints", dict(source=dict(path=str(args.source)),
                                          breakpoints=[dict(line=line)]))["breakpoints"]
                if not breakpoints or not breakpoints[0].get("verified"):
                    raise RuntimeError(f"Unresolved source breakpoint: {breakpoints}")
                dap.request("configurationDone", {})
                dap.response(launch)
                stopped = dap.event("stopped")
                if stopped.get("reason") != "breakpoint":
                    raise RuntimeError(f"Stopped for a reason other than breakpoint: {stopped}")
                thread = stopped["threadId"]
                frame = dap.request("stackTrace", dict(threadId=thread, startFrame=0, levels=1))["stackFrames"][0]
                if not same_source(frame.get("source", {}).get("path", ""), args.source, args.source_map) or frame["line"] != line:
                    raise RuntimeError(f"Stopped at the wrong source location: {frame}")
                scopes = dap.request("scopes", dict(frameId=frame["id"]))["scopes"]
                variables = []
                for scope in scopes:
                    if not scope.get("expensive"):
                        variables.extend(dap.request("variables", dict(variablesReference=scope["variablesReference"]))["variables"])
                dap.request("continue", dict(threadId=thread))
                exited = dap.event("exited")
                if exited["exitCode"] != 0:
                    raise RuntimeError(f"Debuggee failed after breakpoint: {exited}")
                dap.request("disconnect", dict(terminateDebuggee=True))
                return dict(backend="codelldb-dap-tcp", debugger=str(adapter),
                            executable=str(args.executable), fixture=str(args.fixture) if args.fixture else None,
                            breakpoint=breakpoints[0], stopped=stopped, frame=frame,
                            locals=[v for v in variables if v["name"] in ("argc", "inputs", "cfg")],
                            debuggee_exit=exited)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            if args.log and dap:
                args.log.write_text(json.dumps(dap.transcript, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cli", "dap"), default="auto")
    parser.add_argument("--extension", type=Path, default=EXTENSION)
    parser.add_argument("--executable", type=Path, default=ROOT / "build/debug/ptm_debug.exe")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "debug/main.cpp")
    parser.add_argument("--line", type=int, help="Default: source line containing const auto result=ptm::match")
    parser.add_argument("--source-map", nargs=2, action="append", default=[], metavar=("BUILD_PREFIX", "LOCAL_PREFIX"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    args.extension = args.extension.resolve()
    args.executable = args.executable.resolve()
    args.source = args.source.resolve()
    if args.fixture:
        args.fixture = args.fixture.resolve()
    if args.log:
        args.log = args.log.resolve()
    try:
        for path in (args.executable, args.source, *([args.fixture] if args.fixture else [])):
            if not path.is_file():
                raise FileNotFoundError(path)
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        line = args.line or next(index for index, text in enumerate(
            args.source.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"\bconst\s+auto\s+result\s*=\s*ptm::match\s*\(",text))
        errors = []
        backends = [verify_cli, verify_dap] if args.backend == "auto" else [verify_cli if args.backend == "cli" else verify_dap]
        for backend in backends:
            try:
                result = backend(args, line)
                result["fallback_blockers"] = errors
                print(json.dumps(dict(verified=True, **result), indent=2))
                return 0
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                errors.append(f"{backend.__name__}: {error}")
        raise RuntimeError("\n".join(errors))
    except (OSError, RuntimeError, ValueError, StopIteration) as error:
        print(json.dumps(dict(verified=False, blocker=str(error)), indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
