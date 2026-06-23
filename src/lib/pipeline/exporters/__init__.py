"""Dataset export helpers for pipeline outputs."""


def build_webdataset_main():
    """Lazy wrapper for the WebDataset builder entrypoint."""
    from .webdataset import main

    return main()

__all__ = ["build_webdataset_main"]
