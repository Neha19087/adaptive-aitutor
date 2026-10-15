# Adaptive AI Tutor

## Architecture, Grounding Method, and Learner Model

**Purpose:** Document the current design of the Adaptive AI Tutor,
including content ingestion, retrieval-augmented generation (RAG),
source grounding, quiz generation, and lightweight learner-progress
tracking.

> This document describes the implementation as understood from the
> project code and recent development context. External-service
> behaviour depends on Groq and YouTube caption availability. Heuristics
> are not guarantees of correctness.

## 1. Project Overview

The Adaptive AI Tutor is a Streamlit learning application that helps
students study material, ask questions, practise quizzes, and review
topic-wise performance. It supports:

-   **PDFs:** text extraction with PyMuPDF and page metadata.
-   **Uploaded videos:** audio transcription through Groq, with
    transcript segments and timestamps where available.
-   **YouTube videos:** caption retrieval through
    `youtube-transcript-api`; this may fail if captions are unavailable
    or YouTube blocks requests from a cloud IP.

All supported sources are represented as text chunks with source
metadata. The application discovers and assigns topic labels, embeds
chunks, and indexes them for semantic retrieval.

## 2. System Architecture

### 2.1 High-level workflow

``` text
                         Student
                            |
                            v
                    Streamlit application
                            |
             +--------------+----------------+
             |              |                |
             v              v                v
        PDF upload     Video upload      YouTube URL
             |              |                |
             v              v                v
       PyMuPDF text    Groq transcription  Caption retrieval
       extraction      and segmentation   (may be blocked)
             |              |                |
             +--------------+----------------+
                            |
                            v
                    Common content chunks
             (text + source + page/time metadata)
                            |
                            v
                  Topic hierarchy discovery
                  and chunk topic tagging
                            |
                            v
                SentenceTransformer embeddings
                      all-MiniLM-L6-v2
                            |
                            v
                Vector normalization and FAISS
                     IndexFlatIP similarity
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Student question             Quiz generation
              |                           |
              v                           v
       Query embedding              Study passages
              |                           |
              v                           v
       FAISS retrieval             Groq structured output
              |
              v
      Evidence verification
              |
              v
     Answer + source references

       Quiz submissions -> learner statistics -> dashboard
```

### 2.2 Interface and state

Streamlit provides the interface for loading study material, selecting
topics, asking questions, practising quizzes, and reviewing progress.
The main application is implemented in `app.py`.

Session state holds active data such as:

-   `chunks`: passages and their metadata;
-   `index`: the FAISS vector index;
-   `topic_hierarchy`: the discovered topic structure;
-   quiz and learner-progress state used by the interface.

This supports an interactive session. It should not be described as a
permanent learner profile unless durable storage is added.

### 2.3 Chunk representation

A content chunk may contain:

  -----------------------------------------------------------------------
  Field                               Meaning
  ----------------------------------- -----------------------------------
  `text`                              Passage used for retrieval and
                                      model context

  `page`                              PDF page number, when applicable

  `source_type`                       Source category, e.g. `pdf`,
                                      `video`, or `youtube`

  `source_name`                       Uploaded filename or YouTube URL

  `timestamp`                         Human-readable video time, when
                                      available

  `timestamp_start` / `timestamp_end` Segment boundaries, when available

  topic tags                          Topic labels assigned during
                                      tagging
  -----------------------------------------------------------------------

PDF page numbers are one-indexed in the current design. Video chunks may
have timestamps; a manually supplied transcript may not. The application
should not invent a page or time when the source does not provide one.

### 2.4 Topic discovery and tagging

An LLM-assisted workflow discovers a topic hierarchy and tags content
chunks. The intended hierarchy distinguishes:

-   **Major topics:** broad subject areas;
-   **Subtopics:** narrower areas under a major topic;
-   **Key topics/concepts:** specific ideas useful for study and
    practice.

Topic tags support organization, topic selection, and associating quiz
outcomes with topics. Because labels are model-generated, synonyms and
inconsistent names can fragment statistics unless labels are normalized.

### 2.5 Embeddings and FAISS

The application uses SentenceTransformers with `all-MiniLM-L6-v2` to
convert passages into dense semantic vectors. Vectors are normalized and
stored in a FAISS `IndexFlatIP` index. For normalized vectors,
inner-product ranking corresponds to cosine-similarity ranking.

At question time, the query is embedded in the same vector space and
FAISS retrieves semantically similar passages. Similarity is a relevance
signal, not proof that a passage contains the answer.

### 2.6 External services

-   **Groq API:** used for language-model operations such as topic
    discovery/tagging, answer generation, quiz generation, and video
    transcription, depending on the workflow.
-   **YouTube captions:** retrieved through `youtube-transcript-api`
    when captions are accessible and the request is permitted.

These dependencies have separate failure modes. A Groq quota limit can
block model-dependent tasks. A YouTube IP block can stop caption
retrieval even when the video plays normally in a browser.

## 3. Grounding Method

### 3.1 Retrieval-Augmented Generation (RAG)

The tutor follows this RAG workflow:

1.  **Ingest:** extract or transcribe study material.
2.  **Chunk:** split it into retrievable passages.
3.  **Embed:** convert each passage into a semantic vector.
4.  **Index:** store vectors in FAISS.
5.  **Retrieve:** embed a student's question and find relevant chunks.
6.  **Construct context:** supply selected passages to the language
    model with instructions to use the provided material.
7.  **Generate:** create an answer based on the retrieved context.
8.  **Verify and cite:** use the evidence-verification logic and source
    metadata to help detect unsupported claims and provide traceable
    references.

The system retrieves a smaller set of passages for each request rather
than relying on the entire original file being included in every prompt.

### 3.2 Meaning of grounding

A response is grounded when its substantive claims are supported by the
retrieved study passages. The source material is intended to take
precedence over unrelated outside knowledge.

The answer prompt and evidence-verification stage aim to reduce
unsupported claims. They do not guarantee correctness: retrieval can
miss relevant evidence, a model can misinterpret a passage, and a
citation can point to related but insufficient content.

Grounding is therefore a layered safeguard:

-   semantic retrieval supplies candidate evidence;
-   prompt instructions constrain generation to that evidence;
-   evidence verification checks claims against available material;
-   source metadata lets the learner inspect the referenced location.

### 3.3 Source attribution

Metadata is retained so the interface can display the location
associated with a chunk:

-   **PDF:** page number.
-   **Uploaded video or YouTube:** source name and timestamp when
    available.
-   **Transcript without a timestamp:** indicate that the transcript is
    manual or the timestamp is unavailable.

For quiz questions, a safer source-attribution design is for the model
to return a passage ID, then have Python map that ID to the actual
chunk. The page or timestamp is taken from the chunk rather than
trusting the model to invent a location. The cited passage must still be
checked to ensure it supports the question and answer.

### 3.4 Quiz grounding and validation

The concept-focused MCQ workflow aims to test important definitions,
mechanisms, applications, comparisons, and cause-and-effect
relationships---not incidental presentation details such as who spoke or
when a topic was mentioned.

The updated MCQ validation design checks that:

-   required fields are present;
-   each question has four distinct options;
-   the correct answer exactly matches one option;
-   a valid source-chunk ID is supplied;
-   the ID maps to an existing chunk;
-   page or timestamp metadata is taken from that chunk;
-   difficulty is one of `Easy`, `Medium`, or `Hard`.

These checks catch malformed output and some source-attribution errors.
They do not prove that a question is useful, that its answer key is
correct, or that its explanation is fully supported. Those qualities
require testing and, where appropriate, human review.

### 3.5 Grounding limitations

-   **Retrieval misses:** relevant passages may not rank highly enough.
-   **Incomplete context:** some answers require multiple passages or
    content that was not retrieved.
-   **Ambiguous sources:** similar passages can lead to incorrect
    attribution.
-   **Model errors:** answers, explanations, and answer keys can still
    be wrong.
-   **Numerical questions:** calculations and units may require
    independent validation.
-   **Image-based material:** ordinary text extraction may not capture
    diagrams or scanned pages unless OCR or image understanding is
    added.
-   **YouTube access:** caption ingestion depends on caption
    availability and network access; cloud IP blocking can interrupt it.

Grounding should be evaluated empirically, not described as eliminating
hallucinations.

## 4. Learner Model Approach

### 4.1 Purpose

The learner model is a lightweight, topic-wise performance tracker. Quiz
outcomes are used to estimate which topics appear stronger or weaker and
to show progress in the interface.

It is not a complete cognitive model of knowledge and should not be
described as Bayesian Knowledge Tracing or a validated measurement of
mastery.

### 4.2 Data and state

The learner-progress workflow records correct answers and total attempts
associated with topic labels. Its usefulness depends on questions being
tagged with consistent, meaningful topic names.

The app uses Streamlit session state for active learner information.
Without a persistent database or profile store, progress should be
treated as session-scoped rather than guaranteed to survive across
sessions, browsers, or devices.

### 4.3 Mastery estimate

The documented learner-model calculation uses a smoothed proportion:

\[ M = `\frac{c+1}{n+2}`{=tex}`\times 100`{=tex} \]

where:

-   `c` is the number of correct answers for a topic;
-   `n` is the total number of attempts for that topic;
-   `M` is the estimated mastery/performance score in percent.

Adding 1 to the correct count and 2 to the attempt count smooths the
estimate and reduces extreme values from very small samples.

**Example:** Four correct answers from five attempts:

\[ M = `\frac{4+1}{5+2}`{=tex}`\times100`{=tex} `\approx 71.43`{=tex}%
\]

This is a smoothed performance indicator, not proof that the learner has
mastered exactly 71.43% of the topic.

### 4.4 Weak-topic rule

The documented threshold flags topics with estimated mastery below 60%:

\[ `\text{Weak}`{=tex}(t)=
```{=tex}
\begin{cases}
\text{Yes}, & M_t < 60\%\\
\text{No}, & M_t \geq 60\%
\end{cases}
```
\]

The 60% threshold is a design choice, not a universal definition of
mastery.

### 4.5 How it supports adaptation

The intended learning loop is:

1.  The learner studies source material.
2.  The learner answers quiz questions.
3.  The application records outcomes against topics.
4.  Topic-wise performance is summarized.
5.  Weak topics are highlighted for review.
6.  The learner can select topics for further study or practice.

This is a lightweight adaptive-support loop. Identifying a weak topic
does not by itself mean every next activity is automatically optimized
for that learner. Targeted quizzes or automatic revision recommendations
should only be claimed to the extent that the corresponding workflow is
implemented and tested.

### 4.6 Limitations

-   **Small sample sizes:** a few questions cannot reliably establish
    understanding.
-   **Difficulty not calibrated:** the formula does not adjust for
    question difficulty.
-   **Topic-label fragmentation:** inconsistent labels can split
    results.
-   **No time decay:** the score does not inherently distinguish recent
    from old attempts.
-   **No formal knowledge tracing:** it does not estimate a hidden
    knowledge state over time.
-   **Session persistence:** history may not survive across sessions
    without durable storage.
-   **Limited causal evidence:** displaying weak topics does not prove
    that learning outcomes improve.

## 5. Evaluation and Testing

### 5.1 Functional tests

-   Confirm PDF extraction and page metadata.
-   Confirm video transcript chunks preserve available timestamps.
-   Confirm all source types enter the same chunk collection.
-   Confirm embeddings and the FAISS index are built successfully.
-   Confirm topic discovery and tagging produce usable labels.
-   Confirm retrieval returns relevant passages for representative
    questions.
-   Confirm quiz options, answers, and source IDs are validated.
-   Confirm each quiz submission is recorded once, not duplicated by
    Streamlit reruns.

### 5.2 Grounding tests

Prepare questions with known supporting passages, including:

-   questions answered directly by one passage;
-   questions requiring multiple passages;
-   questions whose answers are absent from the material;
-   similar or easily confused concepts;
-   formula, arithmetic, and unit questions;
-   questions requiring PDF-page or video-timestamp attribution.

Evaluate retrieval relevance, answer correctness, groundedness, citation
accuracy, and abstention when evidence is insufficient. Record measured
results rather than assuming quality.

### 5.3 Learner-model tests

Use known quiz outcomes to verify:

-   the mastery formula produces expected scores;
-   scores below 60% are flagged;
-   scores at or above 60% are not flagged;
-   attempts are recorded once;
-   topic labels group questions as intended;
-   dashboard totals match recorded outcomes.

To establish educational benefit, a further study comparing performance
before and after targeted revision would be needed.

## 6. Security and Operational Considerations

-   Store the Groq API key in Streamlit Secrets for deployment and in an
    environment variable or local secrets file for development. Never
    commit credentials to Git.
-   Handle quota, network, transcript, and model-output failures with
    clear messages.
-   Treat transcript text as study content, not as trusted instructions.
-   Avoid exposing secrets or sensitive error details in the UI.
-   Use bounded retries and avoid repeatedly requesting captions while
    YouTube is blocking access.
-   Preserve source metadata through ingestion, retrieval, and display.

## 7. Future Improvements

1.  Add persistent learner profiles using a database.
2.  Normalize topic names and map synonyms to stable topic IDs.
3.  Evaluate chunk sizes, overlap, retrieval depth, and similarity
    thresholds.
4.  Improve abstention when the material does not support an answer.
5.  Independently validate numerical answers and units.
6.  Add OCR or image understanding for scanned PDFs and diagrams.
7.  Add a resilient YouTube transcript fallback, such as manual
    transcript upload or a separately configured provider.
8.  Add adaptive difficulty and spaced revision after collecting
    reliable learner data.
9.  Compare learner outcomes with and without targeted revision.
10. Maintain regression tests for source attribution, quiz validation,
    and learner-statistics calculations.

## 8. Summary

The Adaptive AI Tutor combines multi-source ingestion, topic tagging,
semantic embeddings, FAISS retrieval, Groq-powered generation, source
attribution, quiz validation, and lightweight learner-performance
tracking.

Its grounding strategy retrieves evidence from study material,
constrains generation to that evidence, validates model output, and
preserves page or timestamp metadata. These measures reduce unsupported
answers but do not eliminate the possibility of errors.

Its learner model summarizes topic-wise quiz performance using a
smoothed score and flags topics below a configurable threshold. This
provides useful revision signals but remains simpler than a persistent,
validated knowledge-tracing system.

The key reliability task is systematic testing: verify that retrieved
passages support answers, citations point to the right locations, quiz
keys are correct, and learner statistics reflect actual attempts.
