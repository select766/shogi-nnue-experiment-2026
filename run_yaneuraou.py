#!/usr/bin/env python3
"""YaneuraOu USI interaction script that properly waits for responses."""

import subprocess
import sys
import time


def main():
    engine_path = sys.argv[1] if len(sys.argv) > 1 else "./YaneuraOu-by-gcc"

    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def send(cmd):
        print(f">>> {cmd}")
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def wait_for(keyword):
        while True:
            line = proc.stdout.readline()
            if not line:
                print("<<< (EOF)")
                return None
            line = line.rstrip("\n")
            print(f"<<< {line}")
            if keyword in line:
                return line

    # USI handshake
    send("usi")
    wait_for("usiok")

    # Initialize
    send("isready")
    wait_for("readyok")

    # Test 1: startpos, go byoyomi 1000
    send("position startpos")
    send("go byoyomi 1000")
    bestmove_line = wait_for("bestmove")

    # Test 2: startpos moves 2g2f, go byoyomi 1000
    send("position startpos moves 2g2f")
    send("go byoyomi 1000")
    bestmove_line2 = wait_for("bestmove")

    send("quit")
    proc.wait(timeout=5)


if __name__ == "__main__":
    main()
