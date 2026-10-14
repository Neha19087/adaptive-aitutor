# Adaptive AI Tutor

## System Architecture, Grounding Method, and Learner Model

## 1. Project Overview

The Adaptive AI Tutor is an AI-powered learning application designed to
help students understand study material, explore topics, practise
questions, and monitor their learning progress. It uses uploaded
educational documents as its knowledge source and combines semantic
search with a large language model to provide context-aware answers.

The system aims to make learning more interactive by connecting
explanations and quiz questions to the original study material. It also
maintains a learner model that estimates topic-wise mastery from quiz
performance and identifies topics that may require additional revision.

### Main Objectives

-   Enable students to upload educational material in PDF format.
-   Extract text and organize it into searchable chunks.
-   Retrieve relevant passages when a student asks a question.
-   Generate answers grounded in the supplied material.
-   Display source-page references to support verification.
-   Generate quizzes and provide answers, explanations, and feedback.
-   Track learner performance and highlight potentially weak topics.

------------------------------------------------------------------------

# 2. System Architecture

## 2.1 Architecture Overview

The application follows a modular architecture consisting of document
processing, semantic indexing, retrieval, evidence verification,
AI-generated responses, quiz generation, and learner-progress tracking.

### High-Level Workflow

``` text
             Student
                |
                v
       Streamlit User Interface
                |
        +-------+--------+
        |                |
        v                v
   PDF Upload       Student Question
        |                |
        v                v
   PDF Text        Query Embedding
   Extraction            |
        |                v
        v           FAISS Search
   Text Chunking          |
        |                v
        v          Relevant Chunks
   Embeddings             |
        |                v
        v         Evidence Verification
   FAISS Index             |
                           v
                    Groq LLM API
                           |
                           v
                  Grounded Answer
                  with Page Citations

        Quiz Generation and Assessment
                       |
                       v
                 Quiz Results
                       |
                       v
                  Learner Model
                       |
                       v
             Mastery and Weak Topics
```

The document-ingestion workflow prepares the knowledge base. The
question-answering workflow uses that knowledge base to retrieve
relevant evidence before generating an answer. Quiz results feed into
the learner model to support progress tracking.

## 2.2 Technology Stack

  -----------------------------------------------------------------------
  Component               Technology              Purpose
  ----------------------- ----------------------- -----------------------
  User interface          Streamlit               Provides the
                                                  interactive web
                                                  application

  PDF processing          PyMuPDF                 Extracts text and page
                                                  information from PDFs

  Text embeddings         Sentence Transformers,  Converts text into
                          `all-MiniLM-L6-v2`      numerical vectors

  Vector search           FAISS, `IndexFlatIP`    Retrieves semantically
                                                  similar text chunks

  Language model          Groq API with           Generates answers and
                          `openai/gpt-oss-120b`   structured quiz content

  Application logic       Python                  Coordinates processing,
                                                  retrieval, generation,
                                                  and scoring

  Learner state           Streamlit session state Stores quiz performance
                                                  and progress during the
                                                  session
  -----------------------------------------------------------------------

## 2.3 Document Ingestion and Preprocessing

When a student uploads a PDF, the application extracts its text using
PyMuPDF. The extracted text retains page information so that retrieved
passages can be connected to their original pages.

The extracted text is divided into smaller chunks. Chunking makes the
material easier to search and helps prevent the language model from
receiving an entire document when only a few passages are relevant.

Each chunk contains information such as:

-   The extracted text.
-   The source page number.
-   Topic-related metadata, where available.

The chunks are subsequently converted into embeddings.

## 2.4 Embedding Generation and Indexing

An embedding represents a text passage as a numerical vector. The
application uses the `all-MiniLM-L6-v2` Sentence Transformer model to
generate these vectors.

The vectors are stored in a FAISS `IndexFlatIP` index. The application
can then compare a question embedding against the indexed chunk
embeddings to retrieve potentially relevant passages.

The similarity search helps retrieve passages based on meaning rather
than requiring an exact keyword match.

## 2.5 Topic Hierarchy

The application also uses AI-assisted topic discovery to organize the
study material into major topics, subtopics, and key concepts.

This hierarchy supports topic exploration and quiz selection. It also
provides a basis for organizing learner progress by topic.

Because topic names are generated from the material, the same concept
may occasionally receive slightly different labels. This can affect how
consistently progress is grouped.

## 2.6 Question-Answering Workflow

When a student asks a question, the application follows this sequence:

1.  Convert the question into an embedding.
2.  Search the FAISS index for relevant chunks.
3.  Select the highest-ranking passages that satisfy the retrieval
    criteria.
4.  Check whether the retrieved evidence is sufficient to answer the
    question.
5.  Send the question and retrieved evidence to the Groq language model.
6.  Generate an answer with source-page references.
7.  Return the answer to the student through the Streamlit interface.

The current retrieval approach uses a small set of top-ranked passages,
with a similarity threshold, rather than passing the entire document to
the model.

## 2.7 Quiz-Generation Workflow

The quiz component uses the selected topic and retrieved study material
to generate questions. The balanced-quiz design requests:

-   Three multiple-choice questions.
-   One short-answer question.
-   One numerical question, provided a valid calculation is supported by
    the material.

The questions include relevant metadata such as topic, difficulty,
explanation, and source page. Multiple-choice questions contain options
and a correct answer. Short-answer questions include a reference answer,
while numerical questions include an expected value and tolerance.

The application validates the generated question count, types, and
required fields before accepting the quiz. Consequently, an AI response
that does not match the expected structure can cause quiz generation to
fail.

------------------------------------------------------------------------

# 3. Grounding Method

## 3.1 What Grounding Means

Grounding is the process of constraining an AI-generated answer using
evidence from an identified knowledge source.

In this application, the uploaded study material acts as the primary
knowledge source. Instead of asking the language model to answer solely
from its general training, the application retrieves relevant passages
and supplies them as context.

The objective is to improve factual consistency, reduce unsupported
claims, and help students verify explanations against their original
material.

## 3.2 Retrieval-Augmented Generation (RAG)

The application follows a Retrieval-Augmented Generation (RAG) approach.

RAG combines two processes:

-   **Retrieval:** Finds relevant information in the indexed study
    material.
-   **Generation:** Uses a language model to produce a readable response
    based on the retrieved information.

The application performs semantic retrieval using embeddings and FAISS
before calling the Groq API.

### Grounding Pipeline

``` text
Student Question
       |
       v
Question Embedding
       |
       v
Semantic Search in FAISS
       |
       v
Top-Ranked Study Passages
       |
       v
Similarity Threshold
       |
       v
Evidence Sufficiency Check
       |
       +---- Insufficient evidence
       |             |
       |             v
       |      Decline to answer
       |
       +---- Sufficient evidence
                     |
                     v
             Groq Language Model
                     |
                     v
          Answer with Page References
```

## 3.3 Semantic Retrieval

The application represents both the student's question and the stored
text chunks as vectors. FAISS uses inner-product similarity to rank
candidate passages.

The current implementation uses a top-three retrieval setting and a
similarity threshold of approximately `0.15`.

The threshold is intended to filter out passages that are insufficiently
similar to the question. Its effectiveness depends on the embedding
model, text content, and score distribution, so it should be evaluated
against representative questions rather than treated as a guarantee of
relevance.

## 3.4 Evidence Verification

After retrieval, the application uses an evidence-verification step to
assess whether the selected passages adequately support an answer.

The verification logic distinguishes between:

-   **SUPPORTED:** The retrieved material provides sufficient support
    for answering.
-   **INSUFFICIENT:** The retrieved material does not provide adequate
    evidence.

If the evidence is insufficient, the application is designed to avoid
presenting an unsupported answer as though it were established by the
uploaded material.

This is an important safeguard because a language model can produce a
plausible response even when the retrieved passages do not contain the
required information.

However, the verification step is itself AI-assisted. It can make
mistakes, so it should be considered a safeguard rather than a formal
proof that every generated claim is correct.

## 3.5 Source-Page Citations

The application preserves page numbers during PDF processing. These page
numbers travel with the text chunks and are used when displaying answers
and quiz explanations.

Page references allow students to return to the original document and
check the supporting information.

For example, an answer may contain a reference in the form `[Page 4]`.

A page citation indicates where supporting material was retrieved; it
does not independently prove that every statement in the answer is
correct. Citation accuracy and whether the cited passage actually
supports the claim should be included in evaluation.

## 3.6 Grounding in Quiz Generation

Quiz generation also uses retrieved study material as its context. The
model is instructed to avoid inventing facts, formulas, values, or
answers that are not supported by the source.

Each question is expected to include its source page and an explanation.
The application validates required fields before displaying the quiz.

For numerical questions, the expected answer and tolerance support
automatic comparison with the student's response. The calculation itself
should still be checked against the source material, because
structurally valid generated questions can contain incorrect answers.

## 3.7 Limitations of the Grounding Method

The current approach has several limitations:

1.  **PDF text extraction:** Scanned PDFs and image-only diagrams may
    not provide extractable text without OCR or dedicated image
    processing.
2.  **Retrieval misses:** Relevant information may not appear in the
    top-ranked passages.
3.  **Similarity threshold:** A fixed threshold may reject useful
    passages or accept irrelevant ones.
4.  **Evidence verification:** The verification model may incorrectly
    judge whether evidence is sufficient.
5.  **Citation limitations:** A page number can be present even when the
    cited passage does not fully support the generated claim.
6.  **Numerical correctness:** AI-generated calculations and answer keys
    require validation.
7.  **Multimodal limitations:** Text extraction and semantic search
    alone do not establish that diagrams, images, or other non-textual
    content have been interpreted correctly.

These limitations should be considered when evaluating the reliability
of the tutor.

------------------------------------------------------------------------

# 4. Learner Model Approach

## 4.1 Purpose

The learner model maintains a lightweight estimate of the student's
performance across topics. Its purpose is to help students identify
areas where they may need more practice.

The model records quiz attempts and correct answers by topic. It then
calculates an estimated mastery score and flags topics whose estimated
mastery falls below a defined threshold.

The current approach is a simple, smoothed performance model. It is not
a full Bayesian Knowledge Tracing model.

## 4.2 Learner Data

For each topic, the application maintains two principal values:

  Variable     Meaning
  ------------ ---------------------------------------------------
  `attempts`   Number of quiz questions attempted for that topic
  `correct`    Number of correct answers recorded for that topic

A simplified learner-state representation is:

``` python
learner_model = {
    "Topic A": {
        "attempts": 5,
        "correct": 4
    },
    "Topic B": {
        "attempts": 3,
        "correct": 1
    }
}
```

This example illustrates the structure of the learner data; it is not a
record of an actual student's performance.

The application updates the learner state when a quiz is submitted. A
guard prevents the same submitted quiz from being recorded multiple
times during the same quiz session.

## 4.3 Mastery Estimation

The application uses a smoothed estimate based on the number of correct
answers and total attempts.

Let:

-   \(c\) = number of correct answers.
-   \(n\) = total number of attempts.

The mastery score is calculated as:

\[ M = `\frac{c+1}{n+2}`{=tex}`\times100`{=tex} \]

The addition of 1 to the correct count and 2 to the attempt count
provides a simple smoothing effect. It prevents the estimated mastery
from being exactly 0% or 100% solely because of a very small number of
attempts.

For example, suppose a student attempts five questions and answers four
correctly:

\[ M = `\frac{4+1}{5+2}`{=tex}`\times100`{=tex} \]

\[ M `\approx 71.43`{=tex}% \]

The resulting value is an estimated performance score, not a direct
measurement of the student's complete understanding.

## 4.4 Weak-Topic Identification

The application flags topics whose estimated mastery is below 60%.

The decision rule is:

\[ `\text{Weak Topic}`{=tex} =
```{=tex}
\begin{cases}
\text{Yes}, & M < 60\%\\
\text{No}, & M \geq 60\%
\end{cases}
```
\]

For example, if a topic has one correct answer out of three attempts:

\[ M = `\frac{1+1}{3+2}`{=tex}`\times100`{=tex} = 40% \]

The topic is therefore flagged for revision.

This threshold is a configurable design choice rather than a universally
established boundary between mastery and non-mastery.

## 4.5 How the Learner Model Supports Adaptation

The learner model provides information that can support future
personalized learning decisions.

Its current role includes:

1.  Recording performance by topic.
2.  Calculating estimated mastery.
3.  Identifying topics below the mastery threshold.
4.  Displaying learning progress to the student.
5.  Providing information that can inform topic selection for further
    practice.

For example, a student with a lower estimated mastery score in one topic
can use the progress dashboard to identify that topic for revision and
take another quiz.

The existence of a weak-topic score does not, by itself, mean that the
application automatically constructs a complete personalized study plan.
Such behaviour would need to be implemented and tested separately.

## 4.6 State Management and Limitations

The current learner model uses Streamlit session state. This supports
progress tracking during the active application session.

Unless persistent storage is added, the model should not be described as
a permanent learner profile across separate sessions or devices.

Other limitations include:

-   **Limited evidence:** A few quiz questions may not reliably
    represent overall topic mastery.
-   **Question difficulty:** The current formula does not explicitly
    account for differences in difficulty.
-   **Topic fragmentation:** Inconsistent AI-generated topic labels may
    split performance across multiple entries.
-   **No learning-decay model:** The score does not account for how long
    ago a student learned a concept.
-   **No formal knowledge tracing:** The model does not estimate the
    probability of a hidden knowledge state using Bayesian Knowledge
    Tracing.
-   **Limited adaptation:** Identifying a weak topic does not guarantee
    that the next learning activity is automatically optimized for that
    student.

These constraints make the learner model appropriate as a lightweight
progress-tracking mechanism, while leaving room for more advanced
personalization.

------------------------------------------------------------------------

# 5. Evaluation Plan

The system should be evaluated at three levels.

## 5.1 Architecture and Functionality

Check that:

-   PDF text and page numbers are extracted correctly.
-   Text chunks and embeddings are created successfully.
-   Relevant passages are retrieved for representative questions.
-   Topic discovery and topic selection work as expected.
-   The quiz is generated and validated.
-   Quiz results are recorded only once per submission.
-   The progress dashboard displays the correct counts and mastery
    estimates.

## 5.2 Grounding Quality

Prepare a small benchmark containing:

-   Questions directly answered by the uploaded material.
-   Questions requiring information from more than one passage.
-   Questions whose answers are absent from the material.
-   Questions involving formulas or numerical calculations.
-   Questions where similar concepts could lead to retrieval errors.

Evaluate the application using measures such as:

  -----------------------------------------------------------------------
  Measure                             What It Evaluates
  ----------------------------------- -----------------------------------
  Retrieval relevance                 Whether the retrieved passages are
                                      relevant to the question

  Answer correctness                  Whether the generated answer is
                                      factually correct

  Groundedness                        Whether claims are supported by the
                                      retrieved material

  Citation accuracy                   Whether cited pages contain the
                                      supporting evidence

  Abstention quality                  Whether unsupported questions are
                                      correctly declined
  -----------------------------------------------------------------------

The benchmark should contain expected answers and supporting page
references where possible. Results should be measured and reported
rather than assumed.

## 5.3 Learner-Model Evaluation

Test the learner model using known combinations of attempts and correct
answers. Verify that:

-   The mastery formula produces the expected score.
-   Topics below 60% are flagged.
-   Topics at or above 60% are not flagged.
-   Repeated rendering does not duplicate recorded attempts.
-   Performance is grouped under the intended topic names.

To assess whether the model genuinely improves learning, a further study
would be needed---for example, comparing students' performance before
and after targeted revision.

------------------------------------------------------------------------

# 6. Future Improvements

Potential extensions include:

1.  **Persistent learner profiles:** Store progress in a database so it
    remains available across sessions.
2.  **Improved topic normalization:** Merge synonymous topic labels and
    reduce fragmentation in learner statistics.
3.  **Adaptive quiz difficulty:** Adjust question difficulty based on
    previous performance.
4.  **Spaced revision:** Recommend revisiting concepts after suitable
    intervals.
5.  **Stronger numerical validation:** Independently verify generated
    calculations and answer keys.
6.  **Improved retrieval:** Evaluate different chunk sizes, overlap
    settings, and similarity thresholds.
7.  **Diagram and image understanding:** Add OCR or multimodal
    processing for image-based educational content.
8.  **Formal evaluation:** Use a fixed test set and suitable metrics to
    compare retrieval, answer quality, grounding, and abstention
    behaviour.
9.  **More advanced learner modelling:** Explore Bayesian Knowledge
    Tracing or other knowledge-tracing approaches after collecting
    sufficient interaction data.

------------------------------------------------------------------------

# 7. Conclusion

The Adaptive AI Tutor combines semantic retrieval, evidence-aware answer
generation, source-page references, quiz-based assessment, and
lightweight learner progress tracking.

Its grounding approach uses uploaded study material as context for
generating answers, while the evidence-verification stage is designed to
reduce unsupported responses. Its learner model estimates topic-wise
mastery from quiz outcomes and highlights topics that may require
revision.

The current implementation provides a foundation for source-grounded and
performance-aware learning. Its reliability and educational
effectiveness should be established through systematic testing,
especially for answer correctness, citation support, numerical question
validity, and the usefulness of its mastery estimates.
