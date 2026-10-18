"""Named multi-party dialogue rendering and name pass-through tests.

Covers the final format from docs/multi_speaker_design.md:
        \u7528\u6237\u540d\u53ebTim  (“The user's name is Tim”, zh rendering)
    [user]: ...
    [John]: ...

Key points:
- role=user always renders as [user] (even with a name); the point of view stays in the label
- role=assistant + name -> the other real person, rendered as [<name>]
- role=assistant without name -> the AI assistant, rendered as [assistant]
- the name line appears only when role=user carries a name, once per block, language follows content
- without a name the rendering is byte-identical to the historical behaviour
"""

import pytest

from hy_memory.pipelines.base import (
    ChatMessage,
    WriteRequest,
    parse_user_name_line,
    render_user_name_line,
)
from hy_memory.client import HyMemoryClient


class TestSpeakerLabel:
    def test_user_label_never_uses_name(self):
        assert ChatMessage(role="user", content="x").speaker_label() == "user"
        assert ChatMessage(role="user", content="x", name="Tim").speaker_label() == "user"

    def test_named_assistant_is_a_real_person(self):
        m = ChatMessage(role="assistant", content="x", name="John")
        assert m.speaker_label() == "John"

    def test_nameless_assistant_is_the_ai(self):
        assert ChatMessage(role="assistant", content="x").speaker_label() == "assistant"

    def test_other_roles_fall_back_to_role(self):
        assert ChatMessage(role="system", content="x").speaker_label() == "system"
        assert ChatMessage(role="tool", content="x", tool_name="Read").speaker_label() == "tool"


class TestFlatContentUnchangedWithoutName:
    """without a name the output must match the historical behaviour exactly (single-party chat is the main case)"""

    def test_plain_chat_rendering(self):
        req = WriteRequest(messages=[
            ChatMessage(role="user", content="\u6211\u559c\u6b22\u5496\u5561"),  # "I like coffee"
            ChatMessage(role="assistant", content="\u597d\u7684\uff0c\u6211\u8bb0\u4e0b\u4e86"),  # "ok, noted"
        ])
        assert req.get_flat_content() == "[user]: \u6211\u559c\u6b22\u5496\u5561\n[assistant]: \u597d\u7684\uff0c\u6211\u8bb0\u4e0b\u4e86"

    def test_no_name_line_when_user_has_no_name(self):
        req = WriteRequest(messages=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="John here", name="John"),
        ])
        # the peer has a name but the user does not -> no name line is inserted
        assert req.get_flat_content() == "[user]: hi\n[John]: John here"

    def test_content_only_input_untouched(self):
        assert WriteRequest(content="plain text").get_flat_content() == "plain text"


class TestFlatContentWithNames:
    def test_zh_rendering(self):
        req = WriteRequest(messages=[
            ChatMessage(role="user", content="\u6211\u559c\u6b22\u5496\u5561", name="Tim"),
            ChatMessage(role="assistant", content="\u6211\u4e5f\u559c\u6b22\u8336", name="John"),  # "I like tea too"
        ])
        assert req.get_flat_content() == (
            "\u7528\u6237\u540d\u53ebTim\n"
            "[user]: \u6211\u559c\u6b22\u5496\u5561\n"
            "[John]: \u6211\u4e5f\u559c\u6b22\u8336"
        )

    def test_en_rendering(self):
        req = WriteRequest(messages=[
            ChatMessage(role="user", content="I like coffee", name="Tim"),
            ChatMessage(role="assistant", content="I prefer tea", name="John"),
        ])
        assert req.get_flat_content() == (
            "The user's name is Tim\n"
            "[user]: I like coffee\n"
            "[John]: I prefer tea"
        )

    def test_named_user_with_nameless_ai(self):
        req = WriteRequest(messages=[
            ChatMessage(role="user", content="I like coffee", name="Tim"),
            ChatMessage(role="assistant", content="Noted"),
        ])
        assert req.get_flat_content() == (
            "The user's name is Tim\n"
            "[user]: I like coffee\n"
            "[assistant]: Noted"
        )

    def test_name_line_appears_once_across_many_turns(self):
        req = WriteRequest(messages=[
            ChatMessage(role="user", content="a", name="Tim"),
            ChatMessage(role="assistant", content="b", name="John"),
            ChatMessage(role="user", content="c", name="Tim"),
        ])
        out = req.get_flat_content()
        assert out.count("The user's name is Tim") == 1
        assert out.splitlines()[0] == "The user's name is Tim"

    def test_user_name_helper(self):
        req = WriteRequest(messages=[
            ChatMessage(role="assistant", content="b", name="John"),
            ChatMessage(role="user", content="a", name="Tim"),
        ])
        assert req.user_name() == "Tim"
        assert WriteRequest(messages=[ChatMessage(role="user", content="a")]).user_name() is None


class TestNameLineRoundTrip:
    """reading L1_RAW back must recognise the name line (otherwise it looks like a bare message)"""

    @pytest.mark.parametrize("line,expected", [
        ("\u7528\u6237\u540d\u53ebTim", "Tim"),
        ("The user's name is Tim", "Tim"),
        ("\u7528\u6237\u540d\u53eb Mary Jane ", "Mary Jane"),
        ("[user]: hi", None),
        ("[John]: hi", None),
        ("\u6211\u559c\u6b22\u5496\u5561", None),
        ("", None),
    ])
    def test_parse(self, line, expected):
        assert parse_user_name_line(line) == expected

    def test_render_then_parse(self):
        for zh in (True, False):
            line = render_user_name_line("Tim", zh)
            assert parse_user_name_line(line) == "Tim"

    def test_render_empty_name(self):
        assert render_user_name_line("", True) == ""
        assert render_user_name_line("   ", False) == ""

    def test_flat_content_first_line_is_parseable(self):
        req = WriteRequest(messages=[
            ChatMessage(role="user", content="\u6211\u559c\u6b22\u5496\u5561", name="Tim"),
        ])
        first = req.get_flat_content().splitlines()[0]
        assert parse_user_name_line(first) == "Tim"


class TestNormalizeMessageReadsName:
    def test_speaker_name_kept_for_user_and_assistant(self):
        _, msgs, ctype = HyMemoryClient._parse_input([
            {"role": "user", "name": "Tim", "content": "hi"},
            {"role": "assistant", "name": "John", "content": "yo"},
        ])
        assert ctype == "messages"
        assert msgs[0].name == "Tim"
        assert msgs[1].name == "John"
        # the speaker name must not end up in tool_name
        assert msgs[0].tool_name is None and msgs[1].tool_name is None

    def test_tool_message_name_is_still_the_tool_name(self):
        _, msgs, _ = HyMemoryClient._parse_input([
            {"role": "tool", "tool_call_id": "t1", "name": "Read", "content": "42 lines"},
        ])
        assert msgs[0].tool_name == "Read"
        assert msgs[0].name is None

    def test_anthropic_blocks_keep_speaker_name(self):
        _, msgs, _ = HyMemoryClient._parse_input([
            {"role": "assistant", "name": "John",
             "content": [{"type": "text", "text": "hello"}]},
        ])
        assert msgs[0].name == "John" and msgs[0].content == "hello"

    def test_to_dict_omits_name_when_absent(self):
        assert ChatMessage(role="user", content="hi").to_dict() == {
            "role": "user", "content": "hi",
        }

    def test_to_dict_from_dict_round_trip(self):
        d = ChatMessage(role="assistant", content="yo", name="John").to_dict()
        assert d["name"] == "John"
        assert ChatMessage.from_dict(d).name == "John"
