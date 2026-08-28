"""One-shot refresh of the dynamic persona + capabilities overlays.

Standalone entry point for environments where nexi runs on-demand rather than
as a persistent FastAPI service (e.g. node-a runs only the xnch gateway, so the
``lifespan``-driven ``_capability_refresh_loop`` never fires and the generated
``~/.xnch/nexi-persona.generated.yaml`` overlay would never be written).

Run manually or from a scheduler (systemd timer / cron / periodtask):

    python -m nexi.character.refresh [--verbose]

Writes:
- ~/.xnch/nexi-persona.generated.yaml        (live self-model)
- ~/.xnch/nexi-capabilities.generated.yaml   (live tool/host/status overlay)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from nexi.character import persona_builder
from nexi.character.capability_builder import refresh as refresh_capabilities

logger = logging.getLogger(__name__)


async def arun(verbose: bool = False) -> int:
    """Refresh capability + persona overlays; return process exit code (0 = ok)."""
    cap_result = await refresh_capabilities()
    persona_result = await persona_builder.build_persona()

    errors: list[str] = []
    if cap_result.error:
        errors.append(f"capabilities: {cap_result.error}")
    if persona_result.error:
        errors.append(f"persona: {persona_result.error}")

    if verbose or errors:
        print(f"capabilities overlay changed={cap_result.changed}")
        print(f"persona overlay changed={persona_result.changed}")
        if errors:
            for err in errors:
                print(f"ERROR {err}", file=sys.stderr)

    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh nexi persona + capabilities overlays.")
    parser.add_argument("--verbose", action="store_true", help="print overlay change status")
    args = parser.parse_args()
    return asyncio.run(arun(verbose=args.verbose))


if __name__ == "__main__":
    raise SystemExit(main())
