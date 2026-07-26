"""Notices go to stderr, once, so a sweep over configurations stays readable."""

from __future__ import annotations

import pytest

from dimma.datasets._attribution import emit_once, reset_emitted


@pytest.fixture(autouse=True)
def clean_slate():
    reset_emitted()
    yield
    reset_emitted()


def test_message_goes_to_stderr_not_stdout(capsys):
    emit_once("k", "a notice")
    captured = capsys.readouterr()
    assert "a notice" in captured.err
    assert captured.out == ""


def test_second_call_with_the_same_key_is_silent(capsys):
    emit_once("k", "a notice")
    capsys.readouterr()
    emit_once("k", "a notice")
    assert capsys.readouterr().err == ""


def test_distinct_keys_each_print(capsys):
    emit_once("one", "first")
    emit_once("two", "second")
    err = capsys.readouterr().err
    assert "first" in err
    assert "second" in err


def test_reset_lets_a_key_print_again(capsys):
    emit_once("k", "a notice")
    capsys.readouterr()
    reset_emitted()
    emit_once("k", "a notice")
    assert "a notice" in capsys.readouterr().err
