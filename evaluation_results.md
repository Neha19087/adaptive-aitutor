# Adaptive AI Tutor --- Evaluation Results

## 1. Evaluation Overview

This section evaluates the Adaptive AI Tutor from two perspectives:

1.  **Framework-based RAG evaluation:** A structured evaluation plan
    using metrics commonly associated with RAGAS-style evaluation.
2.  **Simulated student evaluation:** An illustrative
    learner-performance analysis based on the team's 11-page
    network/protocol study material.

The study material includes concepts such as captured traffic, the
fourth byte, message-type bytes, proxy fragmentation, and credential
impact.

> **Reporting note:** The topic-level scores listed in Section 3 were
> observed in the app during a quiz run. The framework-based metric
> values in Section 2 are explicitly marked as simulated examples for
> report planning; they were not produced by an executed RAGAS,
> DeepEval, or TruLens benchmark. Do not present them as measured
> experimental results.

------------------------------------------------------------------------

## 2. Framework-Based Evaluation

### 2.1 Metrics

The following metrics are suitable for assessing a retrieval-augmented
generation (RAG) tutor. RAGAS-style names are used as a reference for
the evaluation design; no framework score is claimed to have been
computed by the current application.

  -----------------------------------------------------------------------
  Metric                  What it evaluates       How to test it
  ----------------------- ----------------------- -----------------------
  Context Precision       Whether retrieved       Review the ranked
                          passages are relevant   passages and mark
                          to the question         relevant versus
                                                  irrelevant chunks

  Context Recall          Whether the retrieved   Compare retrieved
                          passages contain the    chunks with the
                          evidence needed for the passages identified by
                          expected answer         a human evaluator as
                                                  necessary

  Faithfulness /          Whether answer claims   Break answers into
  Groundedness            are supported by the    claims and check each
                          retrieved context       claim against the
                                                  supplied passages

  Answer Relevancy        Whether the response    Have an evaluator rate
                          answers the student's   whether the answer
                          actual question         addresses the question
                                                  directly

  Citation Accuracy       Whether the cited page  Open the cited page and
                          actually supports the   verify the claim
                          associated claim        against the source

  Abstention Quality      Whether the tutor       Ask questions that are
                          declines to answer when intentionally
                          the source lacks        unanswerable from the
                          sufficient evidence     uploaded notes

  Quiz Validity           Whether quiz questions, Compare each generated
                          keys, explanations, and item with the source
                          page references are     and independently check
                          correct                 its answer
  -----------------------------------------------------------------------

### 2.2 Illustrative Simulated Results

The following table is an **illustrative example only** showing how
results could be reported after a small manual benchmark. These values
are not measured results from the app and should be replaced with
results from actual tests before final submission.

  ------------------------------------------------------------------------
  Metric                Illustrative simulated value Interpretation if
                                                     confirmed by testing
  --------------------- ---------------------------- ---------------------
  Context Precision                             0.82 Most retrieved
                                                     passages are relevant

  Context Recall                                0.78 Retrieved passages
                                                     cover most required
                                                     evidence

  Faithfulness /                                0.85 Most answer claims
  Groundedness                                       are supported by the
                                                     retrieved context

  Answer Relevancy                              0.88 Responses generally
                                                     address the question
                                                     asked

  Citation Accuracy                             0.80 Most checked
                                                     citations point to
                                                     supporting pages

  Abstention Accuracy                           0.75 The tutor correctly
                                                     handles most
                                                     unsupported questions

  Quiz Answer-Key                               0.80 Most checked
  Accuracy                                           questions have a
                                                     correct answer key
  ------------------------------------------------------------------------

**Interpretation:** If a real benchmark produced results in this range,
it would suggest that the system is useful but still needs improvement
in retrieval coverage, citations, unsupported-question handling, and
quiz answer verification. These interpretations are conditional; the
illustrative values do not establish that the current system achieves
this performance.

### 2.3 Recommended Benchmark Procedure

Create a small benchmark from the team's course material. A practical
initial set is 20 questions:

-   10 questions directly answerable from one passage.
-   4 questions requiring evidence from more than one passage.
-   3 questions whose answers are absent from the document.
-   3 questions involving protocol details or calculations that require
    careful source verification.

For every question, record:

-   The question.
-   The expected answer.
-   The required supporting page or pages.
-   The retrieved chunks.
-   The generated answer.
-   Whether the answer is supported.
-   Whether the cited page is correct.
-   Whether the system abstained when it should have.

Calculate a metric only after scoring the relevant examples. For
example:

-   **Citation accuracy** = correct supporting citations / citations
    checked.
-   **Abstention accuracy** = correctly handled unsupported questions /
    unsupported questions tested.
-   **Quiz answer-key accuracy** = independently verified correct answer
    keys / answer keys checked.

For RAGAS or another evaluation framework, use the framework's own
required input format and document the model and configuration used.
Metric definitions and aggregation procedures can differ between
framework versions.

------------------------------------------------------------------------

## 3. Observed Application Results

During the app run, a five-question quiz was generated and the
quiz-scoring interface displayed per-answer explanations. The learner
dashboard showed the following topic-level results:

  Topic shown in dashboard      Correct / attempts   Displayed score
  --------------------------- -------------------- -----------------
  Proxy fragmentation                        1 / 1              100%
  Message type byte                          0 / 1                0%
  Credential impact                          0 / 1                0%
  Fragmentation observation                  0 / 1                0%

These are the recorded dashboard observations from the test run, not a
large-scale evaluation.

### 3.1 Interpretation

-   **Proxy fragmentation:** The learner answered the one recorded
    question correctly.
-   **Message type byte:** The one recorded answer was incorrect,
    suggesting that this concept may need review.
-   **Credential impact:** The one recorded answer was incorrect,
    suggesting a possible learning gap.
-   **Fragmentation observation:** The one recorded answer was
    incorrect, suggesting a possible learning gap.

Each topic has only one recorded attempt. Therefore, these scores are
preliminary indicators and are not sufficient to conclude that the
learner has mastered or failed to master a topic.

The dashboard's displayed percentages may represent raw accuracy. If the
implementation also uses the smoothed mastery formula described in the
learner-model documentation, report the raw accuracy and smoothed
estimate separately to avoid confusing the two.

------------------------------------------------------------------------

## 4. Simulated Student Results

The following is a **simulated example** showing how the learner model
could be evaluated with hypothetical students using the same course
material. These are not actual student records or results from a
completed experiment.

### 4.1 Hypothetical student performance

  ----------------------------------------------------------------------------------
  Simulated             Proxy Message type   Credential   Fragmentation        Total
  learner       fragmentation         byte       impact     observation      correct
  ----------- --------------- ------------ ------------ --------------- ------------
  Learner A               1/1          1/1          1/1             1/1   4/4 (100%)

  Learner B               1/1          0/1          0/1             0/1    1/4 (25%)

  Learner C               1/1          1/1          0/1             1/1    3/4 (75%)
  ----------------------------------------------------------------------------------

In this hypothetical illustration, Learner A answers all four topic
questions correctly, Learner B answers one correctly, and Learner C
answers three correctly. The small four-question set demonstrates how
topic-level performance can be represented, but it is too small to make
reliable claims about learning outcomes.

### 4.2 Example of smoothed mastery

The app's documented smoothed mastery estimate is:

\[ M = `\frac{c+1}{n+2}`{=tex}`\times100`{=tex} \]

where (c) is the number of correct answers and (n) is the number of
attempts for a topic.

For example, for a hypothetical topic with 1 correct answer out of 1
attempt:

\[ M = `\frac{1+1}{1+2}`{=tex}`\times100`{=tex} `\approx 66.7`{=tex}% \]

For 0 correct answers out of 1 attempt:

\[ M = `\frac{0+1}{1+2}`{=tex}`\times100`{=tex} `\approx 33.3`{=tex}% \]

This differs from raw accuracy, which would be 100% and 0%,
respectively. If the dashboard displays 100% for a topic with one
correct answer out of one attempt, it is likely displaying raw accuracy
or another measure rather than this smoothed mastery estimate. Verify
the actual dashboard calculation before describing it in the final
report.

### 4.3 What the simulated results demonstrate

The hypothetical results illustrate how the tutor could:

-   Track quiz performance by topic.
-   Distinguish stronger and weaker topics.
-   Highlight concepts for further revision.
-   Update topic estimates as more attempts are recorded.

They do **not** demonstrate that the tutor improves exam performance,
that its quiz questions are always correct, or that its mastery
estimates are educationally validated. Those claims require additional
testing with verified questions and, for learning-effectiveness claims,
a suitable study design.

------------------------------------------------------------------------

## 5. Functional Checks Observed

The following checks were reported as working during the app test:

  -----------------------------------------------------------------------
  Check                               Observed outcome
  ----------------------------------- -----------------------------------
  Quiz generation                     A five-question quiz was displayed
                                      in a test run

  Answer submission and scoring       Worked during the observed test

  Per-answer explanations and source  Displayed in the quiz interface
  pages                               

  Learner progress dashboard          Displayed topic-level performance

  Weak-topic tracking                 Progress data was available for
                                      identifying topics needing review
  -----------------------------------------------------------------------

These are functional observations rather than a formal accuracy
benchmark. They do not independently establish the correctness of every
generated answer, the relevance of every retrieved passage, or the
accuracy of every citation.

------------------------------------------------------------------------

## 6. Limitations

1.  **Small sample size:** The observed topic scores are based on one
    attempt per topic.
2.  **No completed framework benchmark:** RAGAS, DeepEval, or TruLens
    metrics have not yet been measured for this run.
3.  **AI-generated content:** Quiz questions and answer keys can contain
    mistakes and require independent verification.
4.  **Citation verification:** Displaying a page number does not
    guarantee that the cited page supports the complete answer.
5.  **Limited learner evidence:** A small quiz cannot reliably establish
    a student's overall knowledge of a topic.
6.  **Simulated learners:** Hypothetical learner rows are useful for
    demonstrating the evaluation format but must not be represented as
    real participants.
7.  **Metric variability:** Scores depend on the questions, source
    document, retrieval settings, evaluator, and metric implementation.

------------------------------------------------------------------------

## 7. Conclusion

The initial app test demonstrated that the quiz interface, scoring,
explanations, page references, and topic-level progress display can
operate together. The observed dashboard results also show how the
application records performance across concepts from the team's
network/protocol study material.

A formal evaluation remains necessary to quantify retrieval quality,
groundedness, citation accuracy, abstention behaviour, and quiz
answer-key correctness. The illustrative framework scores and simulated
learner rows in this document are examples of reporting format, not
experimentally measured results. Before submission, they should either
be replaced with measured results or retained only when clearly labelled
as simulations.
