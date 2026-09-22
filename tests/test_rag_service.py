from __future__ import annotations

import unittest

from landslide_ai.services.rag_service import get_rag_service


class RAGServiceTest(unittest.TestCase):
    def test_retrieve_returns_guidance_hits(self) -> None:
        service = get_rag_service(".")
        hits = service.retrieve("high landslide risk guidance for terrain driven district", top_k=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertTrue(any("risk" in hit.content.lower() or "monitor" in hit.content.lower() for hit in hits))

    def test_advisory_summary_returns_non_empty_text(self) -> None:
        service = get_rag_service(".")
        summary = service.summarize_for_advisory("moderate landslide risk rainfall guidance", top_k=2)
        self.assertTrue(summary)


if __name__ == "__main__":
    unittest.main()
