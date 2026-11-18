"""serve/test_web_follow.py - the web chat follows the answer only while the reader has not left (#4014).

The page used to pull the view back to the bottom on every frame when it was within 120 px of it, so one wheel notch
(about 100 px) could not scroll up while an answer streamed. Pins, with node, the page's own rule and that the wheel,
touch and keyboard listeners are wired.  Skipped without node.

    python -m unittest serve.test_web_follow -v
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
class WebFollow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = APP.read_text(encoding="utf-8")
        cls.rule = between(cls.src, "const FOLLOW_SLACK", "function scrollDown")

    def follow(self, distance, was, ms):
        js = self.rule + f"\nconsole.log(JSON.stringify(followAfterScroll({distance}, {json.dumps(was)}, {ms})));"
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_a_wheel_notch_inside_the_old_band_is_not_pulled_back(self):
        self.assertFalse(self.follow(100, True, 20))
        self.assertFalse(self.follow(40, True, 20))

    def box(self, steps):
        """This fork: followText on a stand-in box 420 px tall whose content grows 30 px per text; `steps` is a list
        of reader scrollTop values (None = the reader leaves it). Returns the distance from the end after each step."""
        fn = between(self.src, "function followText", "function updateAssistant")
        js = self.rule + fn + """
const box = {clientHeight: 420, scrollHeight: 420, top: 0, n: 0,          // like a browser: scrollTop stays inside 0..scrollHeight-clientHeight
             get scrollTop() { return this.top; }, set scrollTop(v) { this.top = Math.max(0, Math.min(v, this.scrollHeight - this.clientHeight)); },
             set textContent(t) { this.n++; this.scrollHeight = Math.max(420, this.n * 30); this.scrollTop = this.top; }};
const out = [];
for (const reader of %s) {
  if (reader !== null) box.scrollTop = reader;
  followText(box, "x");
  out.push(box.scrollHeight - box.scrollTop - box.clientHeight);
}
console.log(JSON.stringify(out));""" % json.dumps(steps)
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_the_thinking_box_keeps_its_end_in_view_while_it_grows(self):
        self.assertEqual(self.box([None] * 40), [0] * 40, "past 420 px the end stays in view")

    def test_the_thinking_box_lets_go_when_the_reader_scrolls_up_inside_it(self):
        out = self.box([None] * 30 + [100] + [None] * 5)
        self.assertGreater(out[-1], 24, "scrolled up inside the box: it is not pulled back")
        self.assertIn("followText(body, m.reasoning)", self.src, "the streaming thinking uses it")

    def test_back_at_the_bottom_it_follows_again(self):
        self.assertTrue(self.follow(0, False, 10_000))
        self.assertTrue(self.follow(24, False, 10_000))
        self.assertFalse(self.follow(25, False, 10_000))

    def test_the_page_growing_under_a_following_reader_keeps_following(self):
        self.assertTrue(self.follow(80, True, 5_000))
        self.assertFalse(self.follow(80, False, 5_000))
        self.assertFalse(self.follow(401, True, 5_000))

    def test_the_old_120_px_rule_is_gone_and_the_reader_input_is_wired(self):
        self.assertNotIn("< 120", between(self.src, "function renderChat", "$(\"chat\").addEventListener(\"click\""))
        watch = between(self.src, "(function watchReader()", "})();")
        for event in ("wheel", "touchmove", "pointerdown", "keydown", "scroll"):
            self.assertIn(f'"{event}"', watch)
        # a wheel up on a chat that cannot scroll must not switch following off
        self.assertIn("e.deltaY < 0 && s.scrollTop > 0", watch)


if __name__ == "__main__":
    unittest.main()
