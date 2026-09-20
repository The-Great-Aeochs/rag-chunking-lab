import unittest

from chunking.character import chunk


def page(text):
    return [{"page_content": text, "metadata": {"source": "test"}}]


class CharacterChunkerTests(unittest.TestCase):
    def test_long_block_stays_within_size_and_overlaps(self):
        text = "".join(chr(65 + i % 26) for i in range(2400))
        chunks = chunk(page(text), chunk_size=800, chunk_overlap=80)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c["text"]) <= 800 for c in chunks))
        self.assertEqual(chunks[0]["text"][-80:], chunks[1]["text"][:80])

    def test_paragraphs_keep_all_non_whitespace_text(self):
        text = "a" * 925 + "\n\n" + "b" * 903
        chunks = chunk(page(text), chunk_size=200, chunk_overlap=0)

        self.assertTrue(all(len(c["text"]) <= 200 for c in chunks))
        self.assertEqual("".join(c["text"] for c in chunks), text.replace("\n", ""))

    def test_overlap_equal_to_size_does_not_stall(self):
        chunks = chunk(page("x" * 900), chunk_size=200, chunk_overlap=200)

        self.assertTrue(chunks)
        self.assertTrue(all(len(c["text"]) <= 200 for c in chunks))


if __name__ == "__main__":
    unittest.main()
