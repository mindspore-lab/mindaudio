"""Metric function."""


def wer(ref, hyp):
    r"""
    Word error rate (WER) calculation for one utterance.

    WER defines the distance by the amount of minimum operations
    that has to be done for getting from the reference to the hypothesis. The possible operations are:

    - Deletion: A word was deleted.
    - Insertion: A word was added.
    - Substitution: a word was substituted.

    By these operations, WER is defined as :math:`WER = (D+I+S)/(N)`

    We now use dynamic algorithm to find the minimum edits. Define dp, dp[i][j] means minimum edits between ref[:i] and
    hyp[:j]. The transition equation is thus ``dp[i][j] = dp[i-1][j-1] if ref[i-1]==hyp[j-1] else
    min(dp[i-1][j-1] + 1, dp[i][j-1] + 1, dp[i-1][j] + 1)``. The three terms mean through substitution, insertion,
    and deletion respectively.

    Args:
        ref (list): Reference utterance.
        hyp (list): Hypothesis utterance.

    Returns:
        float, WER.

    Examples:
        >>> wer(["who's", "there"], ["who's"])
        0.5
        >>> wer(["who's", "there"], ["what's", "there"])
        0.5
        >>> wer(["who's", "there"], ["who's", "over", "there"])
        0.5
    """
    if not ref:
        raise ValueError("The reference utterance must not be empty.")
    dp = [[0 for _ in range(len(hyp) + 1)] for j in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                dp[0][j] = j
            elif j == 0:
                dp[i][0] = i

    # computation
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)
    return dp[-1][-1] / len(ref)
