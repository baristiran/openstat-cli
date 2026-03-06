"""Text analysis: TF-IDF, word frequency, topic modeling (LDA)."""

from __future__ import annotations
from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("textfreq", usage="textfreq <col> [--top=20] [--stopwords]")
def cmd_textfreq(session: Session, args: str) -> str:
    """Word frequency analysis for a text column.

    Options:
      --top=<n>       show top N words (default: 20)
      --stopwords     remove common English stopwords
      --min=<n>       minimum word length (default: 2)

    Examples:
      textfreq review_text --top=30 --stopwords
      textfreq title --min=4
    """
    import re
    from collections import Counter
    import polars as pl

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: textfreq <col> [--top=20]"

    col = ca.positional[0]
    top_n = int(ca.options.get("top", 20))
    use_stopwords = "stopwords" in ca.flags
    min_len = int(ca.options.get("min", 2))

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "this", "that", "these", "those", "it", "its",
        "i", "we", "you", "he", "she", "they", "my", "your", "our", "their",
        "not", "no", "as", "if", "so", "than", "then", "when", "where", "how",
        "what", "which", "who",
    }

    try:
        df = session.require_data()
        if col not in df.columns:
            return f"Column not found: {col}"

        texts = df[col].drop_nulls().cast(pl.Utf8).to_list()
        all_words = []
        for text in texts:
            words = re.findall(r"[a-zA-Z]+", text.lower())
            if min_len > 1:
                words = [w for w in words if len(w) >= min_len]
            if use_stopwords:
                words = [w for w in words if w not in STOPWORDS]
            all_words.extend(words)

        if not all_words:
            return "No words found after filtering."

        counter = Counter(all_words)
        most_common = counter.most_common(top_n)

        lines = [f"Word Frequency — {col}  (docs={len(texts)}, unique_words={len(counter)})", ""]
        lines.append(f"  {'Rank':<6} {'Word':<25} {'Count':>8} {'%':>7}")
        lines.append("  " + "-" * 50)
        total = len(all_words)
        for rank, (word, cnt) in enumerate(most_common, 1):
            lines.append(f"  {rank:<6} {word:<25} {cnt:>8,} {100*cnt/total:>6.2f}%")
        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "textfreq")


@command("tfidf", usage="tfidf <col> [--top=20] [--max_features=1000]")
def cmd_tfidf(session: Session, args: str) -> str:
    """TF-IDF analysis: identify most distinctive terms in a text column.

    Options:
      --top=<n>              top N terms by mean TF-IDF score (default: 20)
      --max_features=<n>     vocabulary size limit (default: 1000)
      --ngram_min=<n>        minimum n-gram size (default: 1)
      --ngram_max=<n>        maximum n-gram size (default: 1)

    Examples:
      tfidf review_text --top=30
      tfidf comments --ngram_min=2 --ngram_max=2 --top=15
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return "scikit-learn required. Install: pip install scikit-learn"

    import polars as pl
    import numpy as np

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: tfidf <col> [--top=20]"

    col = ca.positional[0]
    top_n = int(ca.options.get("top", 20))
    max_features = int(ca.options.get("max_features", 1000))
    ngram_min = int(ca.options.get("ngram_min", 1))
    ngram_max = int(ca.options.get("ngram_max", 1))

    try:
        df = session.require_data()
        if col not in df.columns:
            return f"Column not found: {col}"

        texts = df[col].drop_nulls().cast(pl.Utf8).to_list()
        if len(texts) < 2:
            return "Need at least 2 documents for TF-IDF."

        vec = TfidfVectorizer(max_features=max_features, stop_words="english",
                              ngram_range=(ngram_min, ngram_max))
        tfidf_matrix = vec.fit_transform(texts)
        feature_names = vec.get_feature_names_out()
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_idx = mean_scores.argsort()[::-1][:top_n]

        lines = [f"TF-IDF Analysis — {col}  (docs={len(texts)}, vocab={len(feature_names)})", ""]
        lines.append(f"  {'Rank':<6} {'Term':<30} {'Mean TF-IDF':>12}")
        lines.append("  " + "-" * 52)
        for rank, idx in enumerate(top_idx, 1):
            lines.append(f"  {rank:<6} {feature_names[idx]:<30} {mean_scores[idx]:>12.5f}")
        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "tfidf")


@command("lda", usage="lda <col> [--topics=5] [--words=10] [--iter=10]")
def cmd_lda(session: Session, args: str) -> str:
    """Latent Dirichlet Allocation (LDA) topic modeling.

    Options:
      --topics=<n>    number of topics (default: 5)
      --words=<n>     top words per topic to show (default: 10)
      --iter=<n>      max iterations (default: 10)
      --max_features=<n>  vocabulary size (default: 1000)

    Examples:
      lda review_text --topics=5
      lda abstract --topics=8 --words=12
    """
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        return "scikit-learn required. Install: pip install scikit-learn"

    import polars as pl
    import numpy as np

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: lda <col> [--topics=5]"

    col = ca.positional[0]
    n_topics = int(ca.options.get("topics", 5))
    n_words = int(ca.options.get("words", 10))
    n_iter = int(ca.options.get("iter", 10))
    max_features = int(ca.options.get("max_features", 1000))

    try:
        df = session.require_data()
        if col not in df.columns:
            return f"Column not found: {col}"

        texts = df[col].drop_nulls().cast(pl.Utf8).to_list()
        if len(texts) < n_topics:
            return f"Need at least {n_topics} documents (got {len(texts)})."

        vec = CountVectorizer(max_features=max_features, stop_words="english", min_df=2)
        dtm = vec.fit_transform(texts)
        feature_names = vec.get_feature_names_out()

        lda_model = LatentDirichletAllocation(
            n_components=n_topics, max_iter=n_iter, random_state=42
        )
        lda_model.fit(dtm)

        lines = [f"LDA Topic Modeling — {col}  (docs={len(texts)}, topics={n_topics})", ""]
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[::-1][:n_words]
            top_words = [feature_names[i] for i in top_words_idx]
            lines.append(f"  Topic {topic_idx + 1}: {', '.join(top_words)}")
        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "lda")
