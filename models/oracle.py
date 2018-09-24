##########################################################################
# Copyright 2018 Kata.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

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
