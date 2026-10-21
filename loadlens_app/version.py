# SPDX-License-Identifier: Apache-2.0
"""Version of LoadLens: one place for the About endpoint, the sidebar footer and tools/release.py.

No settings import here: the module is read by tools and by the Docker build, which have no configuration.
``tools/release.py --version X.Y.Z`` rewrites the ``__version__`` line; keep it on one line.
"""

from __future__ import annotations

import os
import platform

__version__ = "1.0.0"

# The repository is to be renamed to "loadlens" by the owner: change this one constant then.
SOURCE_URL = "https://github.com/Zagato27/LoadLens"
DOCS_URL = SOURCE_URL + "/blob/main/README.md"
LICENSE = "Apache-2.0"

# Date and short commit hash of the image build (docker build --build-arg LOADLENS_BUILD=2026-10-19-0320474);
# empty when LoadLens runs from a source checkout.
BUILD = os.getenv("LOADLENS_BUILD", "")


def version_info() -> dict[str, str]:
    """Version, build id, Python version and the source URL, as plain strings."""
    return {
        "version": __version__,
        "build": BUILD,
        "python": platform.python_version(),
        "source_url": SOURCE_URL,
    }
