from typing import List

from data import Document
from models import AbstractSummarizer


class OracleSummarizer(AbstractSummarizer):
    """An oracle summarizer that selects sentences based on their true labels."""

    def summarize(self, doc: Document, size: int = 3) -> List[int]:
        """Summarize a document by selecting the first sentences whose true labels are true.

        Args:
            doc (Document): The document to summarize.
            size (int): Maximum number of sentences that the summary should have.

        Returns:
            list: The indices of the extracted sentences that form the summary, sorted
                ascending.
        """
        true_ix = [i for i, sent in enumerate(doc.sentences) if sent.label]
        return true_ix[:size]
