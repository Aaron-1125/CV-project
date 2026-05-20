#!/usr/bin/env python3
"""Small hello-world entrypoint for Docker smoke tests."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print("Hello from the ByteDance CV Docker environment.")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")


if __name__ == "__main__":
    main()

