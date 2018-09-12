import abc
import typing

from data import Document


class AbstractSummarizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def summarize(self, doc: Document, size: int = 3) -> typing.List[int]:
        """Summarize a given document.

        Args:
            doc (Document): The document to summarize.
            size (int): Maximum number of sentences that the summary should have.

        Returns:
            list: The indices of the extracted sentences that form the summary, sorted
                ascending.
        """
        pass
