"""Small LLM summarization scaffold.

This is a placeholder that will call an LLM summarizer (local or API) to
produce compressed summaries for training/labeling. Implement connectors
to your preferred LLM provider here.
"""

from typing import List


def summarize_texts(texts: List[str]) -> List[str]:
    # Placeholder: return first 100 chars as a "summary"
    return [t[:100] for t in texts]


if __name__ == '__main__':
    sample = ["This is a long example text about device usage and screen time."]
    print(summarize_texts(sample))
