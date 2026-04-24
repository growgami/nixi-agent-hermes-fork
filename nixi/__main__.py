"""Allow running nixi as ``python -m nixi``.

Container entry point — validates env, seeds config, starts gateway.
See ``nixi.deploy.start_nixi()`` for details.
"""

from nixi.deploy import start_nixi

if __name__ == "__main__":
    start_nixi()