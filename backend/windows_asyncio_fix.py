import sys
import asyncio


def apply_windows_asyncio_policy() -> None:
    """
    Playwright launches a driver subprocess.

    On Windows, asyncio subprocess support requires the Proactor event loop.
    If another package or server setup switches to SelectorEventLoopPolicy,
    sync_playwright() can fail with:

        NotImplementedError in asyncio.base_events._make_subprocess_transport
    """
    if not sys.platform.startswith("win"):
        return

    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        # Defensive fallback for unusual Python builds.
        pass