"""R-29 leftover: the Cursor default root path must not be hardcoded to macOS.

`registry._cursor_data_root` already splits by platform into darwin / win32 / XDG, but
`cursor.cursor_default_root` still returns
`~/Library/Application Support/Cursor/User` hard-coded. That path does not exist on Windows,
so whoever calls it gets a root whose detection always fails.

The two places must hold one opinion: there may only be one fallback rule.
"""

from pathlib import Path

from one_memory.imports.sources.cursor import cursor_default_root
from one_memory.imports.sources.registry import _cursor_data_root


def test_cursor_default_root_on_windows_uses_appdata(tmp_path: Path) -> None:
    home = tmp_path / "home"
    app_data = tmp_path / "Roaming"

    assert (
        cursor_default_root(
            home,
            platform="win32",
            environment={"APPDATA": str(app_data)},
        )
        == app_data / "Cursor" / "User"
    )


def test_cursor_default_root_on_windows_falls_back_when_appdata_is_missing(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"

    assert (
        cursor_default_root(
            home,
            platform="win32",
            environment={},
        )
        == home / "AppData" / "Roaming" / "Cursor" / "User"
    )


def test_cursor_default_root_on_linux_uses_xdg(tmp_path: Path) -> None:
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"

    assert (
        cursor_default_root(
            home,
            platform="linux",
            environment={"XDG_CONFIG_HOME": str(xdg)},
        )
        == xdg / "Cursor" / "User"
    )


def test_cursor_default_root_on_darwin_keeps_the_library_path(tmp_path: Path) -> None:
    """Green control: the macOS behavior must not be changed in passing."""
    home = tmp_path / "home"

    assert cursor_default_root(home, platform="darwin", environment={}) == (
        home / "Library" / "Application Support" / "Cursor" / "User"
    )


def test_registry_fallback_and_cursor_default_root_are_the_same_opinion(
    tmp_path: Path,
) -> None:
    """The two places each had their own copy; once they drifted they will drift again.

    When `~/.cursor` does not exist, the registry fallback must be exactly the path that \
cursor_default_root
    returns -- otherwise on Windows the catalog points at one place while other code reads another.
    """
    home = tmp_path / "home"
    app_data = tmp_path / "Roaming"
    environment = {"APPDATA": str(app_data)}

    from_registry = _cursor_data_root(
        home=home,
        environment=environment,
        platform="win32",
    )
    from_cursor = cursor_default_root(
        home,
        platform="win32",
        environment=environment,
    )
    assert from_registry == from_cursor
