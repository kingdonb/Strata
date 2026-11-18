"""serve/test_web_thinking_rounds.py - the web chat draws thinking after a tool call below that call (this fork).

The thinking of one answer comes in rounds. Before, all of it went into the one box at the top of the answer, so
thinking after a tool call streamed out of view, above the tool rows. Pins, with node, how the page splits m.reasoning
at the tools' `rat` marks. Skipped without node.

    python -m unittest serve.test_web_thinking_rounds -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent / "web" / "app.js"


def between(src: str, start: str, end: str) -> str:
    a = src.index(start)
    return src[a:src.index(end, a)]


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class ThinkingRounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = APP.read_text(encoding="utf-8")
        cls.fns = between(cls.src, "function splitThinking", "function roundThinkingHtml")

    def split(self, m):
        js = self.fns + f"""
const m = {json.dumps(m)};
console.log(JSON.stringify({{first: firstThinking(m), rounds: (m.tools || []).map((_, k) => roundThinking(m, k))}}));"""
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_no_tools_keeps_all_thinking_at_the_top(self):
        self.assertEqual(self.split({"reasoning": "abc", "tools": []}), {"first": "abc", "rounds": []})

    def test_each_round_goes_below_the_last_call_of_the_round_before_it(self):
        m = {"reasoning": "one\n\ntwo\n\nthree", "tools": [{"rat": 3}, {"rat": 3}, {"rat": 8}]}
        self.assertEqual(self.split(m), {"first": "one", "rounds": [None, "two", "three"]},
                         "two calls in one round: the thinking that follows them comes after the second")

    def test_a_round_with_no_thinking_draws_no_box(self):
        m = {"reasoning": "one", "tools": [{"rat": 3}]}
        self.assertEqual(self.split(m)["rounds"], [""])

    def test_a_chat_saved_without_marks_keeps_the_old_layout(self):
        m = {"reasoning": "one two", "tools": [{"at": 0}]}
        self.assertEqual(self.split(m), {"first": "one two", "rounds": [None]})

    def test_the_page_uses_it(self):
        self.assertIn("toolImages(t) + roundThinkingHtml(m, k, streaming)", self.src)
        self.assertIn("followText(body, first)", self.src)
        self.assertIn('details.tool-call, details.think-round', self.src, "patchAnswer keeps a round's box while it streams")


if __name__ == "__main__":
    unittest.main()
