import json
import time

from kunji import keystore, usage
from kunji.catalog import Catalog, fmt_money, normalize
from kunji.config import Config, cache_dir, coerce, config_path


def test_config_roundtrip_and_defaults():
    cfg = Config.load()
    assert cfg.get("display.currency") == "USD"
    cfg.set("display.currency", "INR")
    cfg.set("budget.daily.openai", 200)
    cfg.save()
    again = Config.load()
    assert again.get("display.currency") == "INR"
    assert again.get("budget.daily.openai") == 200
    assert again.get("updates.check") is True  # default still merged in
    assert "sk-" not in config_path().read_text()


def test_coerce():
    assert coerce("true") is True and coerce("off") is False
    assert coerce("3") == 3 and coerce("2.5") == 2.5
    assert coerce('["a:b"]') == ["a:b"]
    assert coerce("hello") == "hello"


def test_keys_env_wins_over_keychain(monkeypatch):
    keystore.set_key("openai", "sk-from-keychain-1234")
    assert keystore.get_key("openai", ("OPENAI_API_KEY",)) == ("sk-from-keychain-1234", "keychain")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-12345678")
    assert keystore.get_key("openai", ("OPENAI_API_KEY",))[1] == "env:OPENAI_API_KEY"
    assert keystore.delete_key("openai")
    assert not keystore.delete_key("openai")


def test_mask_and_redact():
    assert keystore.mask("sk-1234567890abcd") == "sk-123••••••••abcd"
    assert keystore.mask("short") == "••••••••"
    secret = "sk-1234567890abcd"
    assert secret not in keystore.redact(f"bad key {secret}", secret)


def test_normalize():
    assert normalize("claude-sonnet-4.5-20250929") == normalize("claude-sonnet-4-5")
    assert normalize("gpt-4o-2024-08-06") == "gpt-4o"


def _write_cache(name, prices):
    path = cache_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_at": time.time(), "prices": prices}))


def test_price_lookup_order():
    _write_cache(
        "openrouter_prices.json",
        {"openai/gpt-x": [1.0, 2.0], "anthropic/claude-z-4.5": [3.0, 15.0]},
    )
    _write_cache("kunji_prices.json", {"groq:llama": [0.5, 0.5]})
    cfg = Config(path=cache_dir() / "test_config.toml", data={"network": {"catalog": True}})
    cat = Catalog(cfg)
    assert cat.price_for("openai", "gpt-x", "openai").source == "openrouter"
    assert cat.price_for("anthropic", "claude-z-4-5-20250101", "anthropic").input == 3.0
    assert cat.price_for("groq", "llama", None).source == "kunji catalog"
    assert cat.cost("openai", "gpt-x", "openai", 1_000_000, 500_000) == 2.0
    cfg.set("pricing", {"openai:gpt-x": {"input": 9, "output": 9}})
    assert Catalog(cfg).price_for("openai", "gpt-x", "openai").source == "your config"


def test_money_format():
    cfg = Config.load()
    assert fmt_money(cfg, 1.5) == "$1.50"
    assert fmt_money(cfg, 0.0012) == "$0.0012"
    assert fmt_money(cfg, None) == "—"
    cfg.set("display.currency", "INR")
    cfg.set("display.usd_rate", 84)
    assert fmt_money(cfg, 1) == "₹84.00"


def test_budget_states():
    cfg = Config.load()
    cfg.set("budget.daily", {"openai": 1.0})
    assert usage.check_budget(cfg, "openai").state == "ok"
    usage.record("openai", "gpt-x", 10, 10, 0.85)
    assert usage.check_budget(cfg, "openai").state == "warn"
    usage.record("openai", "gpt-x", 10, 10, 0.20)
    assert usage.check_budget(cfg, "openai").state == "block"
    assert usage.check_budget(cfg, "groq").state == "none"
    rows = usage.summary(1)
    assert rows[0].requests == 2 and round(rows[0].cost_usd, 2) == 1.05


def test_budget_exact_thresholds():
    cfg = Config.load()
    cfg.set("budget.daily", {"openai": 1.0})

    usage.record("openai", "gpt-x", 10, 10, 0.8)
    status = usage.check_budget(cfg, "openai")
    assert status.state == "warn"
    assert status.spent == 0.8

    usage.record("openai", "gpt-x", 10, 10, 0.2)
    status = usage.check_budget(cfg, "openai")
    assert status.state == "block"
    assert status.spent == 1.0


def test_budget_without_a_configured_limit_returns_none():
    cfg = Config.load()

    assert usage.check_budget(cfg, "openai").state == "none"


def test_budget_uses_the_most_restrictive_provider_or_all_scope():
    cfg = Config.load()
    cfg.set("budget.daily", {"openai": 1.0, "all": 0.5})
    usage.record("openai", "gpt-x", 10, 10, 0.6)

    # The combined limit is exceeded even though the provider limit only warns.
    status = usage.check_budget(cfg, "openai")
    assert (status.state, status.scope) == ("block", "all")

    cfg.set("budget.daily", {"openai": 0.5, "all": 2.0})
    # A more restrictive provider limit takes precedence over the all-provider limit.
    status = usage.check_budget(cfg, "openai")
    assert (status.state, status.scope) == ("block", "openai")


def test_budget_does_not_count_yesterday_ledger_entries():
    cfg = Config.load()
    cfg.set("budget.daily", {"openai": 1.0})
    path = usage.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    today = time.time()
    yesterday = usage.start_of_day() - 1
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": yesterday,
                        "provider": "openai",
                        "model": "gpt-x",
                        "in": 10,
                        "out": 10,
                        "cost": 50.0,
                        "est": False,
                    }
                ),
                json.dumps(
                    {
                        "ts": today,
                        "provider": "openai",
                        "model": "gpt-x",
                        "in": 10,
                        "out": 10,
                        "cost": 0.25,
                        "est": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status = usage.check_budget(cfg, "openai")
    assert status.state == "ok"
    assert status.spent == 0.25


def test_budget_uses_display_currency_and_usd_rate():
    cfg = Config.load()
    cfg.set("display.currency", "INR")
    cfg.set("display.usd_rate", 84)
    cfg.set("budget.daily", {"openai": 1000})
    usage.record("openai", "gpt-x", 10, 10, 10.0)

    status = usage.check_budget(cfg, "openai")
    assert status.state == "warn"
    assert status.spent == 840.0
    assert status.limit == 1000.0


def test_ledger_fresh_install_no_crash():
    """A fresh install with no usage.jsonl must not crash on any ledger access."""
    path = usage.ledger_path()
    assert not path.exists(), "Ledger should not exist in a fresh install"

    assert usage.entries() == []
    assert usage.entries(since=0.0) == []
    assert usage.spent_today_usd() == 0.0
    assert usage.spent_today_usd(provider="openai") == 0.0
    assert usage.summary(7) == []
    assert usage.summary(1) == []

    cfg = Config.load()
    status = usage.check_budget(cfg, "openai")
    assert status.state == "none"


def test_ledger_empty_file_produces_zero_results():
    """An existing but empty usage.jsonl must behave identically to a missing file."""
    path = usage.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    assert usage.entries() == []
    assert usage.entries(since=0.0) == []
    assert usage.spent_today_usd() == 0.0
    assert usage.summary(7) == []

    cfg = Config.load()
    assert usage.check_budget(cfg, "openai").state == "none"


def test_ledger_with_invalid_lines_is_tolerated():
    """A ledger containing only garbage/invalid
    lines must not crash and must return zero results."""
    path = usage.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json\n\n{broken\n", encoding="utf-8")

    assert usage.entries() == []
    assert usage.spent_today_usd() == 0.0
    assert usage.summary(7) == []
