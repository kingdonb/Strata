"""serve/test_web_tool_rows.py - a tool row in the web chat is not made anew on every token (this fork).

The answer is rewritten on every frame while it streams. With innerHTML every finished tool row was a new element each
time, and its icons flickered. Pins, with node and a small stand-in for the DOM, that patchAnswer keeps a row whose
HTML is unchanged as the same element in the same place, replaces one that changed, and leaves the order right.
Skipped without node.

    python -m unittest serve.test_web_tool_rows -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent / "web" / "app.js"

DOM = r"""
let made = 0;
function el(text) { return {id: ++made, text, parent: null, nextSibling: null,
                            remove() { const p = this.parent; p.kids.splice(p.kids.indexOf(this), 1); p.link(); this.parent = null; this.nextSibling = null; }}; }
function box() { return {kids: [], firstChild: null,
  link() { this.firstChild = this.kids[0] || null; this.kids.forEach((k, i) => { k.parent = this; k.nextSibling = this.kids[i + 1] || null; }); },
  insertBefore(n, ref) { const i = ref ? this.kids.indexOf(ref) : this.kids.length; this.kids.splice(i, 0, n); this.link(); }}; }
// a render: text pieces and tool rows; a tool row is ["tool", html]
function render(b, parts) { patchAnswer(b, parts.map((p) => Array.isArray(p) ? {node: el(p[1]), html: p[1]} : {node: el(p), html: null})); }
const show = (b) => b.kids.map((k) => [k.text, k.id]);
"""


def between(src: str, start: str, end: str) -> str:
    a = src.index(start)
    return src[a:src.index(end, a)]


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class ToolRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = APP.read_text(encoding="utf-8")
        cls.fn = between(src, "function patchAnswer", "function setAnswer")
        cls.src = src

    def run_js(self, body: str):
        out = subprocess.run(["node", "-e", self.fn + DOM + body], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_an_unchanged_row_stays_the_same_element_while_the_text_grows(self):
        a, b = self.run_js("""
            const b = box();
            render(b, ["Let me check.", ["tool", "<row 0 done>"], "The ser"]);
            const first = show(b);
            render(b, ["Let me check.", ["tool", "<row 0 done>"], "The server is"]);
            console.log(JSON.stringify([first, show(b)]));""")
        self.assertEqual([t for t, _ in b], ["Let me check.", "<row 0 done>", "The server is"])
        self.assertEqual(a[1][1], b[1][1], "the tool row is the element it was")
        self.assertNotEqual(a[2][1], b[2][1], "the text after it is new")

    def test_a_row_that_changed_is_replaced_and_the_others_stay(self):
        a, b = self.run_js("""
            const b = box();
            render(b, [["tool", "<row 0 done>"], ["tool", "<row 1 running>"]]);
            const first = show(b);
            render(b, [["tool", "<row 0 done>"], ["tool", "<row 1 done>"], ["tool", "<row 2 running>"], "And"]);
            console.log(JSON.stringify([first, show(b)]));""")
        self.assertEqual([t for t, _ in b], ["<row 0 done>", "<row 1 done>", "<row 2 running>", "And"])
        self.assertEqual(a[0][1], b[0][1])
        self.assertNotEqual(a[1][1], b[1][1])

    def test_nothing_is_left_over_and_an_empty_answer_empties_the_bubble(self):
        a, b = self.run_js("""
            const b = box();
            render(b, ["a", ["tool", "<row 0>"], "b", ["tool", "<row 1>"], "c"]);
            render(b, [["tool", "<row 1>"]]);
            const one = show(b);
            render(b, []);
            console.log(JSON.stringify([one, show(b)]));""")
        self.assertEqual([t for t, _ in a], ["<row 1>"])
        self.assertEqual(b, [])

    def test_icons_point_into_the_page_once_the_sprite_is_copied_in(self):
        fn = between(self.src, "function spriteHref", "function inlineSprite")
        out = subprocess.run(["node", "-e", fn + """
            console.log(JSON.stringify([spriteHref("web/sprite.svg#i-tool", "web/sprite.svg"), spriteHref("#i-tool", "web/sprite.svg"),
                                        spriteHref("other.svg#i-tool", "web/sprite.svg"), spriteHref(null, "web/sprite.svg")]));"""],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), ["#i-tool", "#i-tool", "other.svg#i-tool", None])
        self.assertIn("\ninlineSprite();\nstart();\n", self.src, "the page copies the icons in when it starts")
        self.assertIn('use href="${SPRITE}#i-${name}"', self.src, "a later icon follows SPRITE, which is empty after the copy")

    def test_the_streaming_answer_uses_it(self):
        self.assertIn("setAnswer(bubble, answerHtml(m));", self.src)
        self.assertNotIn("bubble.innerHTML = answerHtml(m)", self.src)


if __name__ == "__main__":
    unittest.main()
