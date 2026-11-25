import pytest

from grammar_fix.config import load_config, parse_hotkey


def test_parse_hotkey():
    hk = parse_hotkey("ctrl+alt+g")
    assert hk.keycode == 5
    assert hk.modifiers == 0x1000 | 0x0800
    assert hk.label == "⌃⌥G"


@pytest.mark.parametrize("combo", ["g", "ctrl+", "hyper+g", "ctrl+f13"])
def test_parse_hotkey_rejects_bad_input(combo):
    with pytest.raises(ValueError):
        parse_hotkey(combo)


def test_load_config_overrides_defaults(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[ollama]\nmodel = "m:1"\n[hotkey]\ncombo = "cmd+shift+k"\n')
    cfg = load_config(path)
    assert cfg.model == "m:1"
    assert cfg.hotkey == "cmd+shift+k"
    assert cfg.max_workers == 4  # default kept
