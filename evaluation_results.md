# Adaptive AI Tutor: Evaluation Report

## 1. Objective

This evaluation plan examines the Adaptive AI Tutor along two dimensions:

1. **Retrieval-augmented generation (RAG):** whether answers are supported by the supplied learning material, whether they address the student's question, and whether retrieval returns relevant evidence.
2. **Personalization:** whether the learner model changes its estimates of topic mastery over multiple practice sessions and how often students encounter repeated questions.

**Important status note:** The numerical values in the results tables below are illustrative, synthetic examples prepared to demonstrate how results can be reported. The RAGAS metrics have **not** been computed by running the project against a completed, verified test set. The personalization figures are from a synthetic simulation, not real student data or a live evaluation of the deployed application. Replace these values with actual outputs before describing them as experimental findings.

## 2. System under evaluation

The tutor uses a retrieval-augmented generation workflow. Learning material is split into text chunks, embedded with a sentence-transformer model, and indexed for similarity search. The application uses a FAISS inner-product index with normalized embeddings, so similarity scores approximate cosine similarity. Retrieved chunks are supplied as context for answer generation.

The source metadata retained by the application supports source labels such as PDF page numbers and video timestamps. The application can process PDF material and video-related content, including transcripts where available. YouTube transcript retrieval can depend on network and hosting conditions, so a transcript failure should be recorded as a source-ingestion limitation rather than treated as a retrieval-quality result.

The learner model tracks topic-level correct answers and attempts. Its estimated mastery is:

`mastery = (correct + 1) / (attempts + 2)`

A topic is classified as weak when its estimated mastery is below `0.60`. This is a simple smoothed estimate: it avoids assigning exactly 0 or 1 mastery from a very small number of attempts. The learner state described here is session-based; it should not be interpreted as persistent long-term learning progress unless persistence has been implemented and verified.

## 3. Evaluation design

### 3.1 RAG test set

A suitable team-built test set should contain questions whose answers can be verified against known source locations. It should also contain questions that cannot be answered from the uploaded material.

Recommended test categories:

- **Direct factual questions:** answerable from one source chunk.
- **Multi-step questions:** require combining evidence from multiple chunks.
- **Off-material questions:** not answerable from the supplied course material.
- **Source-attribution checks:** questions for which the expected PDF page or video timestamp is known.

For every test item, record:

- question;
- tutor-generated answer;
- retrieved context chunks;
- reference answer;
- reference context or source location;
- question type;
- whether the answer is supported by the material.

The placeholder dataset supplied with the evaluation pack must be replaced with real questions, actual outputs from the tutor, and verified source references before it can support a valid evaluation.

### 3.2 Metrics

| Metric | What it checks | Desired interpretation |
|---|---|---|
| Faithfulness | Whether claims in the generated answer are supported by the retrieved context | Higher is better |
| Answer relevancy | Whether the generated answer addresses the user's question | Higher is better |
| Context precision | Whether retrieved chunks are relevant rather than distracting | Higher is better |
| Context recall | Whether the retrieval includes the evidence needed to answer the question | Higher is better |

These metrics are not interchangeable. A relevant answer may still contain unsupported claims, and a faithful answer may be incomplete if retrieval omitted necessary evidence. Off-material questions should additionally be checked for appropriate abstention or a clear statement that the material does not contain the answer.

## 4. RAG evaluation results

### 4.1 Illustrative example results — NOT measured

The following values are **synthetic placeholders for report formatting only**. They are not RAGAS outputs and must not be presented as measured performance.

| Metric | Illustrative value |
|---|---:|
| Faithfulness | 0.88 |
| Answer relevancy | 0.91 |
| Context precision | 0.84 |
| Context recall | 0.86 |

### 4.2 Illustrative interpretation

If a real evaluation produced values in this range, it would suggest that most generated claims were supported by retrieved evidence and that the answers generally addressed the questions. Context precision would be a useful place to investigate irrelevant chunks, while context recall would help identify cases where the retriever failed to return all the evidence required.

These interpretations are conditional examples only. Because the figures above have not been measured, they do **not** establish that the current application achieves this performance.

### 4.3 Off-material questions

Off-material questions are important because a tutor should not confidently invent course-specific facts when the uploaded sources do not contain the answer. During evaluation, record whether the tutor:

1. explicitly acknowledges that the answer is not present in the available material;
2. avoids unsupported claims;
3. distinguishes source-grounded information from general explanation; and
4. avoids presenting a fabricated page number or timestamp.

Report the count and percentage of off-material questions handled appropriately after testing. No measured abstention rate is claimed in this draft.

## 5. Personalization evaluation

### 5.1 Method

A lightweight simulation can test whether the learner model updates its mastery estimates across practice sessions. Use several synthetic student profiles with different initial topic performance, then record answers and mastery after each session.

For each topic, the application uses:

`mastery = (correct + 1) / (attempts + 2)`

The weak-topic threshold is `mastery < 0.60`.

Two useful summary measures are:

- **Mastery gain (percentage points):** final mastery estimate minus initial mastery estimate, multiplied by 100.
- **Question repetition rate:** repeated question presentations divided by all question presentations, multiplied by 100. The implementation must define precisely what counts as a repeat—for example, the same question ID shown to the same profile more than once.

A rise in this smoothed mastery estimate indicates that the simulation recorded more correct answers relative to attempts. It does not, by itself, prove real learning gains or establish that personalization caused the improvement.

### 5.2 Synthetic simulation results

The following are the illustrative outputs from the synthetic profile simulation prepared with the evaluation pack. They are **simulated data**, not observations from real learners and not evidence of a live classroom trial.

| Synthetic profile | Illustrative mastery gain | Illustrative question repetition rate |
|---|---:|---:|
| Profile A | +15.77 percentage points | 41.67% |
| Profile B | +17.38 percentage points | 41.67% |
| Profile C | +9.52 percentage points | 25.00% |

### 5.3 Interpretation of the simulation

- The synthetic profiles show positive changes in their mastery estimates under the simulated answer schedules.
- The profiles have different gains because their simulated starting states and answer histories differ.
- Repetition rates vary across profiles, indicating that repeated-question exposure depends on the question schedule and the definition of a repeated item.
- These figures demonstrate how the learner-model logic and reporting pipeline can be inspected. They do not demonstrate that real students improved, nor do they establish that the tutor's question selection is optimally personalized.

For a stronger experiment, compare the adaptive question schedule against a non-adaptive baseline using the same simulated profiles and question bank. Keep the number of practice opportunities equal, record per-session mastery, and report the question-selection rules and repeat definition.

## 6. Limitations

1. The RAG metric values in Section 4 are illustrative only; a completed RAGAS run has not been verified.
2. The provided test dataset contains placeholder examples until replaced with real, source-verified items.
3. A valid evaluation requires actual tutor answers and actual retrieved chunks, not manually invented contexts.
4. Synthetic profiles are useful for checking software behavior but are not a substitute for a user study.
5. The mastery formula is a simple estimate based on correctness and attempt counts. It does not account for question difficulty, guessing, forgetting, or transfer to new questions.
6. Repetition rate depends on the chosen definition of “repeated question” and should be calculated from a logged sequence of question IDs.
7. Retrieval quality depends on chunking, embedding quality, the material uploaded, and source availability. Transcript-access failures should be reported separately.
8. Source labels such as page numbers and timestamps should be verified against the original material.

## 7. Conclusion

The system has an identifiable evaluation path for both RAG quality and personalization. RAG evaluation should use a verified test set containing answerable, multi-step, source-attribution, and off-material questions. Faithfulness, answer relevancy, context precision, and context recall provide complementary views of generation and retrieval quality.

The learner model provides a transparent baseline for adapting practice using topic-level correctness and attempt counts. Synthetic profile simulations can help verify the update logic and calculate mastery changes and repetition rates. However, the illustrative numbers in this report must not be interpreted as measured system performance. Actual RAGAS metrics require running the evaluation on real tutor outputs and verified references; real personalization claims require appropriately collected evaluation data.

