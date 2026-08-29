from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("requirements", type=Path)
    args = parser.parse_args()

    for raw_line in args.requirements.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        requirement = Requirement(line)
        installed_version = version(requirement.name)
        if not requirement.specifier.contains(installed_version):
            raise RuntimeError(
                f"{requirement.name} {installed_version} does not satisfy {requirement.specifier}"
            )


if __name__ == "__main__":
    main()
