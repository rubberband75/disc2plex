"""Tests for the TMDB client, cache, and show/season resolution. No real network."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from conftest import FIXTURES
from disc2plex import cli
from disc2plex.models import EpisodeInfo, SeasonEpisodes, ShowInfo, ShowSummary
from disc2plex.tmdb import (
    AmbiguousShowError,
    TMDBCache,
    TMDBClient,
    TMDBError,
    TMDBNotFoundError,
    choose_show,
    expected_runtime_seconds,
    parse_season,
    parse_show,
    resolve_show,
    season_episodes,
    select_group_season,
)

DVD_GROUP = "5b0c3a5c0e0a267b7a00a1f1"
PROD_GROUP = "5c1d2e3f4a5b6c7d8e9f0a1b"
FAKE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.fake-test-token"


def tmdb_fixture(name: str) -> dict[str, object]:
    data: dict[str, object] = json.loads((FIXTURES / "tmdb" / f"{name}.json").read_text())
    return data


ROUTES = {
    "/3/search/tv": "search_firefly",
    "/3/tv/1437": "tv_1437",
    "/3/tv/1437/episode_groups": "tv_1437_episode_groups",
    "/3/tv/1437/season/1": "tv_1437_season_1",
    f"/3/tv/episode_group/{DVD_GROUP}": "episode_group_dvd",
}


class FakeTMDB:
    """Records requests and serves fixtures by URL path."""

    def __init__(self, routes: dict[str, str] | None = None) -> None:
        self.routes = dict(ROUTES if routes is None else routes)
        self.requests: list[httpx.Request] = []
        self.responses: dict[str, list[httpx.Response]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if queued := self.responses.get(path):
            return queued.pop(0)
        if path == "/3/search/tv" and request.url.params.get("query") == "Doctor Who":
            return httpx.Response(200, json=tmdb_fixture("search_doctor_who"))
        if path in self.routes:
            return httpx.Response(200, json=tmdb_fixture(self.routes[path]))
        return httpx.Response(404, json={"status_message": "not found"})

    def client(self, cache: TMDBCache | None = None, token: str = FAKE_TOKEN) -> TMDBClient:
        return TMDBClient(
            token, cache=cache, transport=httpx.MockTransport(self.handler), sleep=lambda _s: None
        )

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


@pytest.fixture
def fake() -> FakeTMDB:
    return FakeTMDB()


@pytest.fixture
def firefly(fake: FakeTMDB) -> ShowInfo:
    return fake.client().get_show(1437)


# --- parsing ------------------------------------------------------------------


def test_parse_show() -> None:
    show = parse_show(tmdb_fixture("tv_1437"), tmdb_fixture("tv_1437_episode_groups"))
    assert (show.tmdb_id, show.name, show.year) == (1437, "Firefly", 2002)
    assert [s.number for s in show.seasons] == [0, 1]
    assert show.default_runtime_minutes == 44
    assert [g.type_name for g in show.episode_groups] == ["DVD", "Production"]


def test_parse_show_tolerates_missing_fields() -> None:
    show = parse_show({"id": 5, "name": "X", "first_air_date": "", "episode_run_time": []})
    assert show.year is None
    assert show.default_runtime_minutes is None
    assert show.episode_groups == ()


def test_parse_season_sorts_by_episode_number() -> None:
    episodes = parse_season(tmdb_fixture("tv_1437_season_1"))
    assert [e.episode for e in episodes] == list(range(1, 15))
    assert episodes[0].title == "The Train Job"
    assert episodes[10].runtime_minutes == 86
    assert episodes[0].air_date is not None and episodes[0].air_date.year == 2002


# --- choose_show --------------------------------------------------------------


def _hit(tmdb_id: int, name: str, year: int | None) -> ShowSummary:
    return ShowSummary(tmdb_id=tmdb_id, name=name, year=year)


def test_choose_show_exact_name_wins() -> None:
    results = [_hit(99001, "Firefly Lane", 2021), _hit(1437, "Firefly", 2002)]
    assert choose_show("firefly", results, None).tmdb_id == 1437


def test_choose_show_normalizes_punctuation() -> None:
    results = [
        _hit(1403, "Marvel's Agents of S.H.I.E.L.D.", 2013),
        _hit(2, "Agents of Something", 2015),
    ]
    assert choose_show("marvels agents of shield", results, None).tmdb_id == 1403


def test_choose_show_ambiguous_without_year() -> None:
    results = [_hit(57243, "Doctor Who", 2005), _hit(121, "Doctor Who", 1963)]
    with pytest.raises(AmbiguousShowError) as excinfo:
        choose_show("Doctor Who", results, None)
    assert {c.tmdb_id for c in excinfo.value.candidates} == {57243, 121}
    assert "--tmdb-id" in str(excinfo.value)


def test_choose_show_year_disambiguates() -> None:
    results = [_hit(57243, "Doctor Who", 2005), _hit(121, "Doctor Who", 1963)]
    assert choose_show("Doctor Who", results, 1963).tmdb_id == 121


def test_choose_show_single_fuzzy_result_accepted() -> None:
    assert choose_show("firefly 2002", [_hit(1437, "Firefly", 2002)], None).tmdb_id == 1437


def test_choose_show_no_results() -> None:
    with pytest.raises(TMDBNotFoundError):
        choose_show("Nonexistent", [], None)


# --- select_group_season ------------------------------------------------------


@pytest.mark.parametrize(
    ("groups", "season", "expected"),
    [
        ([{"name": "Season 1", "order": 1}, {"name": "Season 2", "order": 2}], 2, "Season 2"),
        ([{"name": "Specials", "order": 0}, {"name": "Season 1", "order": 1}], 1, "Season 1"),
        ([{"name": "Series 03", "order": 5}], 3, "Series 03"),
        ([{"name": "Volume 1", "order": 0}, {"name": "Volume 2", "order": 1}], 2, "Volume 2"),
        # Unhelpful names: fall back to the Nth non-special group by order.
        ([{"name": "Specials", "order": 0}, {"name": "B", "order": 2}, {"name": "A", "order": 1}],
         2, "B"),
        ([{"name": "Specials", "order": 0}, {"name": "Season 1", "order": 1}], 0, "Specials"),
    ],
)  # fmt: skip
def test_select_group_season(groups: list[dict[str, object]], season: int, expected: str) -> None:
    chosen = select_group_season(groups, season)
    assert chosen is not None and chosen["name"] == expected


def test_select_group_season_missing() -> None:
    assert select_group_season([{"name": "Season 1", "order": 1}], 4) is None
    assert select_group_season([{"name": "Season 1", "order": 1}], 0) is None


def test_season_10_does_not_match_season_1_name() -> None:
    groups = [{"name": "Season 1", "order": 1}, {"name": "Season 10", "order": 10}]
    chosen = select_group_season(groups, 1)
    assert chosen is not None and chosen["name"] == "Season 1"


# --- client: requests, auth, errors -------------------------------------------


def test_bearer_token_sent_as_header(fake: FakeTMDB) -> None:
    fake.client().search_tv("Firefly")
    request = fake.requests[0]
    assert request.headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert "api_key" not in request.url.params
    assert request.url.params["query"] == "Firefly"
    assert request.url.params["language"] == "en-US"


def test_v3_api_key_sent_as_param(fake: FakeTMDB) -> None:
    key = "0123456789abcdef0123456789abcdef"
    fake.client(token=key).search_tv("Firefly", year=2002)
    request = fake.requests[0]
    assert request.url.params["api_key"] == key
    assert request.url.params["first_air_date_year"] == "2002"
    assert "Authorization" not in request.headers


def test_missing_token_gives_clear_error() -> None:
    with pytest.raises(TMDBError, match="TMDB_API_TOKEN"):
        TMDBClient("")


@pytest.mark.parametrize(
    ("status", "error", "match"),
    [
        (401, TMDBError, "rejected the API token"),
        (404, TMDBNotFoundError, "no resource"),
        (500, TMDBError, "HTTP 500"),
    ],
)
def test_http_errors(fake: FakeTMDB, status: int, error: type[Exception], match: str) -> None:
    fake.responses["/3/tv/1437"] = [httpx.Response(status)]
    with pytest.raises(error, match=match):
        fake.client().get("/tv/1437")


def test_rate_limit_retries_then_succeeds(fake: FakeTMDB) -> None:
    sleeps: list[float] = []
    fake.responses["/3/search/tv"] = [
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(429, headers={"Retry-After": "999"}),
    ]
    client = TMDBClient(
        FAKE_TOKEN, transport=httpx.MockTransport(fake.handler), sleep=sleeps.append
    )
    assert client.search_tv("Firefly")[0].tmdb_id == 1437
    assert sleeps == [3.0, 10.0]  # capped


def test_rate_limit_gives_up(fake: FakeTMDB) -> None:
    fake.responses["/3/search/tv"] = [httpx.Response(429)] * 5
    with pytest.raises(TMDBError, match="429"):
        fake.client().search_tv("Firefly")


def test_network_error_wrapped() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    client = TMDBClient(FAKE_TOKEN, transport=httpx.MockTransport(boom))
    with pytest.raises(TMDBError, match="Could not reach TMDB"):
        client.search_tv("Firefly")


def test_invalid_json_wrapped(fake: FakeTMDB) -> None:
    fake.responses["/3/search/tv"] = [httpx.Response(200, content=b"<html>")]
    with pytest.raises(TMDBError, match="invalid JSON"):
        fake.client().search_tv("Firefly")


# --- cache --------------------------------------------------------------------


def test_cache_hit_avoids_request(fake: FakeTMDB, tmp_path: Path) -> None:
    cache = TMDBCache(tmp_path)
    fake.client(cache).get_show(1437)
    count = len(fake.requests)
    fake.client(TMDBCache(tmp_path)).get_show(1437)
    assert len(fake.requests) == count


def test_cache_expires(fake: FakeTMDB, tmp_path: Path) -> None:
    now = [1_000_000.0]
    clock: Callable[[], float] = lambda: now[0]  # noqa: E731
    fake.client(TMDBCache(tmp_path, ttl_seconds=60, clock=clock)).search_tv("Firefly")
    now[0] += 61
    fake.client(TMDBCache(tmp_path, ttl_seconds=60, clock=clock)).search_tv("Firefly")
    assert fake.paths() == ["/3/search/tv", "/3/search/tv"]


def test_cache_keys_by_params(fake: FakeTMDB, tmp_path: Path) -> None:
    client = fake.client(TMDBCache(tmp_path))
    client.search_tv("Firefly")
    client.search_tv("Firefly", year=2002)
    assert len(fake.requests) == 2


def test_cache_never_stores_credentials(fake: FakeTMDB, tmp_path: Path) -> None:
    key = "0123456789abcdef0123456789abcdef"
    fake.client(TMDBCache(tmp_path), token=key).search_tv("Firefly")
    fake.client(TMDBCache(tmp_path)).get_show(1437)
    contents = "".join(p.read_text() for p in tmp_path.glob("*.json"))
    assert key not in contents
    assert FAKE_TOKEN not in contents


def test_cache_ignores_corrupt_file(fake: FakeTMDB, tmp_path: Path) -> None:
    cache = TMDBCache(tmp_path)
    fake.client(cache).search_tv("Firefly")
    for file in tmp_path.glob("*.json"):
        file.write_text("garbage")
    assert fake.client(cache).search_tv("Firefly")[0].tmdb_id == 1437
    assert len(fake.requests) == 2


def test_errors_are_not_cached(fake: FakeTMDB, tmp_path: Path) -> None:
    fake.responses["/3/tv/1437"] = [httpx.Response(500)]
    with pytest.raises(TMDBError):
        fake.client(TMDBCache(tmp_path)).get("/tv/1437")
    assert fake.client(TMDBCache(tmp_path)).get("/tv/1437")["name"] == "Firefly"


# --- resolve_show / season_episodes ---------------------------------------------


def test_resolve_show_by_name(fake: FakeTMDB) -> None:
    show = resolve_show(fake.client(), name="Firefly")
    assert show.tmdb_id == 1437
    assert len(show.episode_groups) == 2


def test_resolve_show_by_id_skips_search(fake: FakeTMDB) -> None:
    resolve_show(fake.client(), name="ignored", tmdb_id=1437)
    assert "/3/search/tv" not in fake.paths()


def test_resolve_show_ambiguous(fake: FakeTMDB) -> None:
    with pytest.raises(AmbiguousShowError):
        resolve_show(fake.client(), name="Doctor Who")


def test_resolve_show_requires_something(fake: FakeTMDB) -> None:
    with pytest.raises(TMDBError):
        resolve_show(fake.client())


def _titles(season: SeasonEpisodes) -> list[str]:
    return [e.title for e in season.episodes]


def test_season_aired_order(fake: FakeTMDB, firefly: ShowInfo) -> None:
    season = season_episodes(fake.client(), firefly, 1, "aired")
    assert _titles(season)[:2] == ["The Train Job", "Bushwhacked"]
    assert season.order == "aired"


def test_season_dvd_order_keeps_aired_numbers(fake: FakeTMDB, firefly: ShowInfo) -> None:
    season = season_episodes(fake.client(), firefly, 1, "dvd")
    assert _titles(season)[:4] == ["Serenity", "The Train Job", "Bushwhacked", "Shindig"]
    first: EpisodeInfo = season.episodes[0]
    assert (first.season, first.episode) == (1, 11)  # named by aired number for Plex
    assert "DVD" in season.label
    assert len(season.episodes) == 14


def test_season_by_group_id(fake: FakeTMDB, firefly: ShowInfo) -> None:
    season = season_episodes(fake.client(), firefly, 1, DVD_GROUP)
    assert _titles(season)[0] == "Serenity"


def test_season_dvd_order_missing_season(fake: FakeTMDB, firefly: ShowInfo) -> None:
    with pytest.raises(TMDBError, match="no season 2"):
        season_episodes(fake.client(), firefly, 2, "dvd")


def test_season_bad_order(fake: FakeTMDB, firefly: ShowInfo) -> None:
    with pytest.raises(TMDBError, match="--order"):
        season_episodes(fake.client(), firefly, 1, "production")


def test_dvd_order_without_dvd_group(fake: FakeTMDB, firefly: ShowInfo) -> None:
    no_groups = firefly.model_copy(update={"episode_groups": ()})
    with pytest.raises(TMDBError, match="no DVD episode group"):
        season_episodes(fake.client(), no_groups, 1, "dvd")


def test_dvd_order_with_several_dvd_groups(fake: FakeTMDB, firefly: ShowInfo) -> None:
    dvd = firefly.episode_groups[0]
    twice = firefly.model_copy(
        update={"episode_groups": (dvd, dvd.model_copy(update={"id": PROD_GROUP}))}
    )
    with pytest.raises(TMDBError, match="several DVD episode groups"):
        season_episodes(fake.client(), twice, 1, "dvd")


def test_expected_runtime(fake: FakeTMDB, firefly: ShowInfo) -> None:
    season = season_episodes(fake.client(), firefly, 1)
    assert expected_runtime_seconds(season) == 44 * 60  # median ignores the 86-min pilot


def test_expected_runtime_falls_back_to_show_default(firefly: ShowInfo) -> None:
    season = SeasonEpisodes(
        show=firefly,
        season=1,
        order="aired",
        label="x",
        episodes=(EpisodeInfo(season=1, episode=1, title="A"),),
    )
    assert expected_runtime_seconds(season) == 44 * 60
    bare = season.model_copy(
        update={"show": firefly.model_copy(update={"default_runtime_minutes": None})}
    )
    assert expected_runtime_seconds(bare) is None


# --- CLI ----------------------------------------------------------------------

runner = CliRunner(env={"COLUMNS": "200"})


@pytest.fixture
def cli_env(fake: FakeTMDB, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeTMDB:
    """Point the CLI at the fake TMDB, an isolated cache, and no real .env files."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("TMDB_API_TOKEN", FAKE_TOKEN)
    real_client = cli.TMDBClient

    def client_factory(token: str, **kwargs: object) -> TMDBClient:
        return real_client(token, transport=httpx.MockTransport(fake.handler), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "TMDBClient", client_factory)
    return fake


def test_cli_tmdb_search(cli_env: FakeTMDB) -> None:
    result = runner.invoke(cli.app, ["tmdb", "search", "Firefly"])
    assert result.exit_code == 0, result.output
    assert "1437" in result.output
    assert "Firefly Lane" in result.output


def test_cli_tmdb_show_lists_groups(cli_env: FakeTMDB) -> None:
    result = runner.invoke(cli.app, ["tmdb", "show", "1437"])
    assert result.exit_code == 0, result.output
    assert DVD_GROUP in result.output
    assert "DVD" in result.output


def test_cli_tmdb_season_dvd(cli_env: FakeTMDB) -> None:
    result = runner.invoke(cli.app, ["tmdb", "season", "1437", "1", "--order", "dvd"])
    assert result.exit_code == 0, result.output
    first_row = next(line for line in result.output.splitlines() if "s01e11" in line)
    assert "Serenity" in first_row
    assert result.output.index("Serenity") < result.output.index("The Train Job")


def test_cli_tmdb_uses_cache_dir(cli_env: FakeTMDB, tmp_path: Path) -> None:
    runner.invoke(cli.app, ["tmdb", "search", "Firefly"])
    assert list((tmp_path / "cache" / "disc2plex" / "tmdb").glob("*.json"))


def test_cli_missing_token(cli_env: FakeTMDB, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TMDB_API_TOKEN")
    result = runner.invoke(cli.app, ["tmdb", "search", "Firefly"])
    assert result.exit_code == 1
    assert "TMDB_API_TOKEN is not set" in result.output


def test_cli_reads_token_from_dotenv(
    cli_env: FakeTMDB, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TMDB_API_TOKEN")
    (tmp_path / ".env").write_text(f"TMDB_API_TOKEN={FAKE_TOKEN}\n")
    result = runner.invoke(cli.app, ["tmdb", "search", "Firefly"])
    assert result.exit_code == 0, result.output
    assert cli_env.requests[0].headers["Authorization"] == f"Bearer {FAKE_TOKEN}"


def test_cli_tmdb_error_is_clean(cli_env: FakeTMDB) -> None:
    result = runner.invoke(cli.app, ["tmdb", "season", "1437", "1", "--order", "nonsense"])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output
