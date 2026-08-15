def clean_text(text):
    return text.strip()


def test_clean_text_removes_outer_whitespace():
    assert clean_text("  hello world  ") == "hello world"