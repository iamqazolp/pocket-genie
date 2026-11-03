from __future__ import annotations

from typing import Any

from pocket_genie.registry.tools import sanitize_json_schema

try:
    from lmformatenforcer import (
        CharacterLevelParserConfig,
        JsonSchemaParser,
        SequenceParser,
        StringParser,
    )
    from lmformatenforcer.consts import COMPLETE_ALPHABET
    from lmformatenforcer.jsonschemaparser import StringParsingState
except ImportError:
    CharacterLevelParserConfig = None  # type: ignore[assignment,misc]
    COMPLETE_ALPHABET = ""
    JsonSchemaParser = None  # type: ignore[assignment,misc]
    SequenceParser = None  # type: ignore[assignment,misc]
    StringParser = None  # type: ignore[assignment,misc]
    StringParsingState = None  # type: ignore[assignment,misc]

_LATIN1_AND_PUNCT: str = (
    "".join(chr(c) for c in range(0xA0, 0x100)) + "–—…’‘“”•€£¥℃™|~♥♠♦♣"
)
COMPACT_ALPHABET: str = COMPLETE_ALPHABET + _LATIN1_AND_PUNCT


def build_tool_call_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct a unified JSON Schema over candidate tools avoiding multi-parser UnionParser blowup."""
    branches: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "unknown")
        params = sanitize_json_schema(fn.get("parameters", {"type": "object", "properties": {}}))
        branches.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": name},
                    "arguments": params,
                },
                "required": ["name", "arguments"],
            }
        )
    if not branches:
        return {"type": "object"}
    if len(branches) == 1:
        return branches[0]
    return {"anyOf": branches}


if JsonSchemaParser is not None and SequenceParser is not None:

    def _upgrade_parser_tree(parser: Any) -> Any:
        if type(parser) is JsonSchemaParser:
            upgraded = FastJsonSchemaParser.__new__(FastJsonSchemaParser)
            upgraded.__dict__.update(parser.__dict__)
            if getattr(upgraded, "context", None) is not None:
                upgraded.context.active_parser = upgraded
            return upgraded
        if type(parser) is SequenceParser:
            parser.parsers = [_upgrade_parser_tree(p) for p in parser.parsers]
            upgraded = FastSequenceParser.__new__(FastSequenceParser)
            upgraded.__dict__.update(parser.__dict__)
            return upgraded
        if hasattr(parser, "parsers") and isinstance(parser.parsers, list):
            parser.parsers = [_upgrade_parser_tree(p) for p in parser.parsers]
        return parser

    class FastJsonSchemaParser(JsonSchemaParser):  # type: ignore[misc]
        """JsonSchemaParser with token-mask caching and open freetext string fast-path."""

        def _in_open_freetext_string(self) -> bool:
            if not self.object_stack:
                return False
            curr = self.object_stack[-1]
            return (
                StringParsingState is not None
                and isinstance(curr, StringParsingState)
                and curr.seen_opening_quote
                and not curr.seen_closing_quote
                and not curr.allowed_strings
                and not curr.regex_parser
                and (curr.max_length is None or len(curr.parsed_string) < curr.max_length)
            )

        def cache_key(self) -> Any:
            if self._in_open_freetext_string():
                curr = self.object_stack[-1]
                return ("json_str_body", (curr.FollowedByChars, self.last_non_whitespace_character))
            return None

        def shortcut_key(self) -> Any:
            if self._in_open_freetext_string():
                curr = self.object_stack[-1]
                if curr.min_length is None or len(curr.parsed_string) >= curr.min_length:
                    return ("json_freetext", "any_char_or_close")
                return ("json_freetext", "any_char_no_close")
            return None

        def add_character(self, new_character: str) -> Any:
            if self._in_open_freetext_string() and new_character not in ('"', "\\"):
                self.context.active_parser = self
                curr = self.object_stack[-1]
                updated_stack = self.object_stack[:-1] + [curr.add_character(new_character)]
                updated = FastJsonSchemaParser(self.context, self.config, updated_stack, 0)
                updated.context.active_parser = updated
                updated.last_parsed_string = self.last_parsed_string
                updated.last_non_whitespace_character = (
                    self.last_non_whitespace_character
                    if new_character in " \t\n\r"
                    else new_character
                )
                return updated
            res = super().add_character(new_character)
            res = _upgrade_parser_tree(res)
            if isinstance(res, FastJsonSchemaParser) and res._in_open_freetext_string():
                res.num_consecutive_whitespaces = 0
            return res

    class FastSequenceParser(SequenceParser):  # type: ignore[misc]
        """SequenceParser forwarding cache_key and shortcut_key while head parser is active."""

        def cache_key(self) -> Any:
            if self.parsers and not self.parsers[0].can_end():
                return self.parsers[0].cache_key()
            return super().cache_key()

        def shortcut_key(self) -> Any:
            if self.parsers and not self.parsers[0].can_end() and hasattr(self.parsers[0], "shortcut_key"):
                return self.parsers[0].shortcut_key()
            return super().shortcut_key()
else:
    FastJsonSchemaParser = None  # type: ignore[assignment,misc]
    FastSequenceParser = None  # type: ignore[assignment,misc]
