"""Saved prompts, chat history and file attachments."""

import json
import subprocess

import httpx
import pytest
from typer.testing import CliRunner

from kunji import attachments, history, prompts
from kunji.cli import _rewrite_prompt_shortcut, app
from kunji.config import Config

runner = CliRunner()


# Prompts --------------------------------------------------------------------


def test_builtins_and_render():
    names = set(prompts.all_prompts())
    assert {"commit", "explain", "review", "fix", "summarize", "translate", "tests"} <= names
    text = prompts.render(prompts.get("@explain"), "def f(): return {1: 2}")
    assert "def f(): return {1: 2}" in text  # braces survive (no str.format)
    assert "{input}" not in text


def test_tests_prompt_in_builtins():
    p = prompts.get("tests")
    assert p.builtin
    assert p.description == "Generate unit tests for the supplied code using pytest"
    text = prompts.render(p, "def add(a, b): return a + b")
    assert "def add(a, b): return a + b" in text
    assert "{input}" not in text
    assert "pytest" in text.lower()


def test_render_requires_input():
    with pytest.raises(prompts.PromptError):
        prompts.render(prompts.get("explain"), "")


def test_user_prompt_overrides_and_parses_description():
    prompts.save("explain", "Explain like I'm five:\n\n{input}", "Simple explain")
    p = prompts.get("explain")
    assert not p.builtin and p.description == "Simple explain"
    assert prompts.render(p, "gravity").startswith("Explain like I'm five")
    assert prompts.remove("explain")
    assert prompts.get("explain").builtin


def test_template_without_input_appends_text():
    prompts.save("tone", "Rewrite politely.")
    assert prompts.render(prompts.get("tone"), "send it now").endswith("send it now")


def test_invalid_names():
    with pytest.raises(prompts.PromptError):
        prompts.save("Bad Name", "x")
    with pytest.raises(prompts.PromptError):
        prompts.get("does-not-exist")


def test_git_diff_in_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "a.txt").write_text("hello\n")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    text = prompts.render(prompts.get("commit"), "", cwd=repo)
    assert "+hello" in text and "```diff" in text


def test_git_diff_outside_repo(tmp_path):
    with pytest.raises(prompts.PromptError):
        prompts.git_diff(tmp_path)


def test_at_shortcut_rewrites_argv():
    assert _rewrite_prompt_shortcut(["kunji", "@commit", "-m", "x"]) == [
        "kunji",
        "run-prompt",
        "commit",
        "-m",
        "x",
    ]
    assert _rewrite_prompt_shortcut(["kunji", "chat"]) == ["kunji", "chat"]


def test_prompts_cli_add_list_remove():
    added = runner.invoke(
        app,
        [
            "prompts",
            "add",
            "standup",
            "--text",
            "Turn into a standup update:\n{input}",
            "-d",
            "Standup",
        ],
    )
    assert added.exit_code == 0
    listed = runner.invoke(app, ["prompts"])
    assert "@standup" in listed.output and "@commit" in listed.output
    assert runner.invoke(app, ["prompts", "remove", "standup"]).exit_code == 0


def test_run_prompt_end_to_end(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        )

    import kunji.engine as engine

    real = engine.make_provider
    monkeypatch.setattr(
        engine,
        "make_provider",
        lambda spec, **kw: real(spec, transport=httpx.MockTransport(handler)),
    )
    cfg = Config.load()
    cfg.set("default_model", "ollama:llama")
    cfg.save()
    result = runner.invoke(app, ["run-prompt", "summarize", "a long text", "--raw"])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output
    assert "a long text" in seen["body"]["messages"][-1]["content"]


# History --------------------------------------------------------------------


def test_history_save_find_search_delete():
    rec = history.new_record("openai:gpt-x")
    rec.messages = [
        {"role": "user", "content": "How do I reverse a list in Python?"},
        {"role": "assistant", "content": "Use reversed() or slicing [::-1]."},
    ]
    history.save(rec)
    assert rec.title.startswith("How do I reverse")
    assert history.find(rec.id[:4]).id == rec.id
    assert history.latest().id == rec.id
    hits = history.search("slicing")
    assert hits and "slicing" in hits[0].snippet
    assert "## You" in history.to_markdown(rec)
    history.delete(rec.id)
    assert history.list_all() == []
    with pytest.raises(history.HistoryError):
        history.find("zzz")


def test_history_cli():
    rec = history.new_record("groq:llama")
    rec.messages = [{"role": "user", "content": "hello there"}]
    history.save(rec)
    listed = runner.invoke(app, ["history"])
    assert rec.id in listed.output
    assert rec.id in runner.invoke(app, ["history", "search", "hello"]).output
    assert runner.invoke(app, ["history", "delete", rec.id]).exit_code == 0


def test_history_limit_rejects_zero_and_negative():
    result = runner.invoke(app, ["history", "--limit", "0"])
    assert result.exit_code == 2
    assert "limit must be greater than 0" in result.output.lower()
    result = runner.invoke(app, ["history", "--limit", "-5"])
    assert result.exit_code == 2
    assert "limit must be greater than 0" in result.output.lower()


def test_chat_session_saves_and_resumes(monkeypatch):
    from kunji.chat import ChatSession

    def handler(request):
        return httpx.Response(
            200, content=b'data: {"choices":[{"delta":{"content":"hi!"}}]}\n\ndata: [DONE]\n\n'
        )

    import kunji.engine as engine

    real = engine.make_provider
    monkeypatch.setattr(
        engine,
        "make_provider",
        lambda spec, **kw: real(spec, transport=httpx.MockTransport(handler)),
    )
    cfg = Config.load()
    cfg.set("default_model", "ollama:llama")
    session = ChatSession(cfg)
    session.send("hello")
    assert session.record is not None
    saved = history.find(session.record.id)
    assert [m["role"] for m in saved.messages] == ["user", "assistant"]

    resumed = ChatSession(cfg, resume=saved)
    assert resumed.ref == "ollama:llama" and len(resumed.messages) == 2
    resumed.send("again")
    assert len(history.find(saved.id).messages) == 4


def test_history_can_be_turned_off(monkeypatch):
    from kunji.chat import ChatSession

    cfg = Config.load()
    cfg.set("default_model", "ollama:llama")
    cfg.set("chat.history", False)
    session = ChatSession(cfg)
    from kunji.providers import Message

    session.messages = [Message("user", "x"), Message("assistant", "y")]
    session.save()
    assert history.list_all() == []


# Attachments ----------------------------------------------------------------


def test_attach_text_file(tmp_path):
    f = tmp_path / "main.py"
    f.write_text("print('hi')\n")
    a = attachments.load(str(f))
    out = attachments.with_files("what does this do?", [a])
    assert "```python" in out and "print('hi')" in out and out.startswith("what does")


def test_attach_rejects_binary_big_missing_and_dirs(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(attachments.AttachmentError):
        attachments.load(str(tmp_path / "x.bin"))
    (tmp_path / "big.txt").write_text("a" * (attachments.MAX_FILE_BYTES + 1))
    with pytest.raises(attachments.AttachmentError):
        attachments.load(str(tmp_path / "big.txt"))
    with pytest.raises(attachments.AttachmentError):
        attachments.load(str(tmp_path / "nope.txt"))
    with pytest.raises(attachments.AttachmentError):
        attachments.load(str(tmp_path))


def test_attach_handles_backticks(tmp_path):
    f = tmp_path / "README.md"
    f.write_text("```bash\necho hi\n```\n")
    out = attachments.render([attachments.load(str(f))])
    assert out.count("````") == 2


def test_chat_file_command(tmp_path, console_at):
    from kunji.chat import ChatSession

    console_at(80)
    cfg = Config.load()
    cfg.set("default_model", "ollama:llama")
    session = ChatSession(cfg)
    f = tmp_path / "notes.txt"
    f.write_text("todo: ship v0.2")
    session.command(f"/file {f}")
    assert len(session.pending) == 1
    session.command("/file clear")
    assert session.pending == []


# Home menu ------------------------------------------------------------------


def test_home_menu_choices_adapt_to_setup():
    from kunji.home import EXIT, Home

    home = Home()
    values = [getattr(c, "value", None) for c in home.main_choices()]
    assert values[0] == "add_provider" and EXIT in values and "chat" not in values
    cfg = Config.load()
    cfg.set("default_model", "ollama:llama")
    cfg.save()
    home = Home()
    values = [getattr(c, "value", None) for c in home.main_choices()]
    assert values[0] == "chat" and "add_provider" not in values
    for value in values:
        if value and value.strip() and value != EXIT:
            assert hasattr(home, f"do_{value}"), value


def test_plain_kunji_without_tty_shows_welcome():
    result = runner.invoke(app, [])
    assert result.exit_code == 0 and "kunji keys add" in result.output
