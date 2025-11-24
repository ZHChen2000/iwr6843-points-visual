# PyInstaller hook for backports module
# This hook file helps PyInstaller collect backports submodules

from PyInstaller.utils.hooks import collect_submodules

# Collect all backports submodules if backports package exists
hiddenimports = []
try:
    import backports
    hiddenimports.extend(collect_submodules('backports'))
except ImportError:
    # If backports is not installed, add common submodules that might be needed
    hiddenimports = [
        'backports.functools_lru_cache',
    ]
except Exception:
    # If collect_submodules fails, use minimal set
    hiddenimports = [
        'backports.functools_lru_cache',
    ]

