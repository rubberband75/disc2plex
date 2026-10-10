"""Tests for the optional Plex scan trigger. No network: a mock transport plays Plex."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from disc2plex.plex import (
    PlexClient,
    PlexError,
    PlexSection,
    find_section,
    parse_sections,
    server_path,
)

SECTIONS_JSON = {
    "MediaContainer": {
        "size": 3,
        "Directory": [
            {"key": "1", "title": "Movies", "type": "movie", "Location": [{"path": "/data"}]},
            {
                "key": "2",
                "title": "TV Shows",
                "type": "show",
                "Location": [{"id": 3, "path": "/data/TV"}, {"id": 4, "path": "/archive/TV"}],
            },
            {
                "key": "5",
                "title": "Kids TV",
                "type": "show",
                "Location": [{"path": "/data/TV/Kids"}],
            },
        ],
    }
}


def test_parse_sections() -> None:
    sections = parse_sections(SECTIONS_JSON)
    assert sections[1] == PlexSection(2, "TV Shows", "show", ("/data/TV", "/archive/TV"))
    assert parse_sections({}) == []


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ("/data/TV/Firefly (2002)", 2),
        ("/archive/TV", 2),
        ("/data/TV/Kids/Bluey (2018)", 5),  # most specific folder wins
        ("/data/TVShows/x", None),  # prefix of a name isn't containment
        ("/data/Movies/x", None),  # movie libraries are ignored
    ],
)
def test_find_section(path: str, key: int | None) -> None:
    found = find_section(parse_sections(SECTIONS_JSON), path)
    assert (found.key if found else None) == key


def test_server_path_mapping() -> None:
    root = Path("/mnt/plex/TV")
    show = root / "Firefly (2002) {tmdb-1437}"
    assert server_path(show, root, None) == str(show)
    assert server_path(show, root, "/data/TV") == "/data/TV/Firefly (2002) {tmdb-1437}"


def client(handler: object) -> PlexClient:
    return PlexClient(
        "http://plex.local:32400/",
        "secret",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


def test_client_lists_sections_and_refreshes() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/library/sections":
            return httpx.Response(200, json=SECTIONS_JSON)
        return httpx.Response(200)

    with client(handler) as plex:
        assert [s.key for s in plex.sections()] == [1, 2, 5]
        plex.refresh(2, "/data/TV/Firefly (2002)")
    assert all(r.headers["X-Plex-Token"] == "secret" for r in seen)
    assert seen[1].url.path == "/library/sections/2/refresh"
    assert seen[1].url.params["path"] == "/data/TV/Firefly (2002)"
    # The token travels in a header, never in the URL (which may end up in logs).
    assert "secret" not in str(seen[1].url)


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(401), "rejected the token"),
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, content=b"<xml/>"), "unexpected response"),
    ],
)
def test_client_errors(response: httpx.Response, match: str) -> None:
    with pytest.raises(PlexError, match=match), client(lambda _r: response) as plex:
        plex.sections()


def test_client_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(PlexError, match="could not reach Plex"), client(handler) as plex:
        plex.refresh(2, "/x")


def test_client_requires_token() -> None:
    with pytest.raises(PlexError, match="PLEX_TOKEN"):
        PlexClient("http://plex.local:32400", "")
