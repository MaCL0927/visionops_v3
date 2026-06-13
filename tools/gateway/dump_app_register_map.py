#!/usr/bin/env python3
"""输出 M6 业务 App 寄存器映射。"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from edge.gateway_adapter.apps.carton_partition_check.register_map import make_register_map as partition_map
from edge.gateway_adapter.apps.carton_tube_check.register_map import make_register_map as tube_map


FACTORIES = {"carton_tube_check": tube_map, "carton_partition_check": partition_map}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="输出 Gateway 业务 App register map")
    parser.add_argument("--app", choices=tuple(FACTORIES), required=True)
    parser.add_argument("--base", type=int)
    args = parser.parse_args(argv)
    definitions = FACTORIES[args.app]() if args.base is None else FACTORIES[args.app](args.base)
    print(json.dumps([
        {"address": item.address, "name": item.name, "type": item.data_type, "scale": item.scale, "description": item.description}
        for item in definitions
    ], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
