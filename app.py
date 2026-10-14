import json
import streamlit as st
import pymupdf
import numpy as np
import faiss
import os
from groq import Groq
from sentence_transformers import SentenceTransformer

def get_weak_topics(learner_model):
    weak_topics = []

    for topic, data in learner_model.items():
        attempts = data["attempts"]
        correct = data["correct"]

        mastery = (correct + 1) / (attempts + 2)

        if mastery < 0.6:
            weak_topics.append(topic)

    return weak_topics

# Initialize the learner model
if "learner_model" not in st.session_state:
    st.session_state["learner_model"] = {}


st.set_page_config(
    page_title="Adaptive AI Tutor",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Adaptive AI Tutor")
st.write("Upload notes and search them using natural language.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def create_chunks(pdf_file, chunk_size=1000, overlap=1):
    document = pymupdf.open(stream=pdf_file, filetype="pdf")
    chunks = []

    for page_number, page in enumerate(document):
        text = page.get_text(sort=True).strip()

        if not text:
            continue

        # Divide text into paragraphs
        paragraphs = text.split("\n\n")

        # Combine paragraphs into chunks
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = " ".join(paragraph.split())

            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "page": page_number + 1
                })

                current_chunk = ""

            current_chunk += " " + paragraph

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "page": page_number + 1
            })

    document.close()
    return chunks


def call_groq_json(system_prompt, user_prompt):
    """Ask Groq for JSON and convert the response to Python data."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    # Remove JSON code fences if present
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
        result = result.rsplit("```", 1)[0]

    return json.loads(result)


def discover_topic_hierarchy(chunks):
    """Find major topics, subtopics and key concepts."""

    # Include page numbers so topics can be traced to sources.
    context = ""

    for i, chunk in enumerate(chunks):
        context += (
            f"\n[Chunk {i} | Page {chunk['page']}]\n"
            f"{chunk['text'][:700]}\n"
        )

    # Keep the discovery request within a reasonable size.
    context = context[:18000]

    system_prompt = """
You organize educational study material into a topic hierarchy.

Use ONLY the supplied material.
Do not invent topics that are not supported by it.

Identify:
1. Major topics: broad subject areas.
2. Subtopics: concepts belonging to a major topic.
3. Key topics: specific concepts, methods, definitions or ideas.

Use consistent names throughout.
Merge duplicate concepts instead of creating duplicate topics.

Return only valid JSON in this format:
{
  "major_topics": [
    {
      "name": "Major topic name",
      "subtopics": [
        {
          "name": "Subtopic name",
          "key_topics": ["Concept A", "Concept B"]
        }
      ]
    }
  ]
}
"""

    user_prompt = (
        "Identify the topic hierarchy in this study material:\n"
        + context
    )

    data = call_groq_json(system_prompt, user_prompt)

    if not isinstance(data.get("major_topics"), list):
        raise ValueError("Invalid topic hierarchy returned by AI.")

    return data["major_topics"]


def tag_content_chunks(chunks, topic_hierarchy):
    """Attach topic metadata to each existing chunk."""

    # Flatten the hierarchy to make tagging easier.
    allowed_topics = []

    for major in topic_hierarchy:
        for subtopic in major.get("subtopics", []):
            allowed_topics.append({
                "major_topic": major["name"],
                "subtopic": subtopic["name"],
                "key_topics": subtopic.get("key_topics", [])
            })

    if not allowed_topics:
        raise ValueError("No usable topics were discovered.")

    # Process eight chunks per request.
    for start in range(0, len(chunks), 8):
        batch = chunks[start:start + 8]

        batch_text = ""

        for i, chunk in enumerate(batch):
            batch_text += (
                f"\nChunk {i}\n"
                f"Page: {chunk['page']}\n"
                f"Text: {chunk['text']}\n"
            )

        system_prompt = """
Tag each content chunk using the supplied topic hierarchy.

Rules:
- Use only topic names from the supplied hierarchy.
- Select the best matching major topic and subtopic.
- Select zero or more matching key topics.
- If no topic fits, use "Unclassified" for major_topic
  and subtopic, and an empty list for key_topics.
- Do not rewrite or remove the original content.
- Return one result for every input chunk, in the same order.

Return only valid JSON:
{
  "tags": [
    {
      "chunk_index": 0,
      "major_topic": "Topic name",
      "subtopic": "Subtopic name",
      "key_topics": ["Concept A"]
    }
  ]
}
"""

        user_prompt = (
            "Allowed topic hierarchy:\n"
            + json.dumps(topic_hierarchy, ensure_ascii=False)
            + "\n\nContent chunks:\n"
            + batch_text
        )

        data = call_groq_json(system_prompt, user_prompt)
        tags = data.get("tags", [])

        # Apply tags only when the response matches the input chunk.
        for tag in tags:
            i = tag.get("chunk_index")

            if not isinstance(i, int) or not 0 <= i < len(batch):
                continue

            chunk = batch[i]

            chunk["major_topic"] = tag.get(
                "major_topic", "Unclassified"
            )
            chunk["subtopic"] = tag.get(
                "subtopic", "Unclassified"
            )
            chunk["key_topics"] = tag.get(
                "key_topics", []
            )

    # Ensure no chunk is missing metadata.
    for chunk in chunks:
        chunk.setdefault("major_topic", "Unclassified")
        chunk.setdefault("subtopic", "Unclassified")
        chunk.setdefault("key_topics", [])

    return chunks

def generate_quiz(chunks, number_of_questions=5, focus_topic=None):
    context = ""

    for chunk in chunks:
        context += (
            f"\n[Page {chunk['page']}]\n"
            f"{chunk['text']}\n"
        )

    system_prompt = """
You are a careful quiz creator.

Use ONLY the supplied study material. Do not invent facts,
formulas, numerical values, or answers.

Create exactly 5 questions:
- 3 MCQs
- 1 short-answer question
- 1 numerical question, only if a calculation is supported
  by the supplied material.

If the material does not contain a ready-made numerical
example, create a simple calculation using only formulas,
facts, and values explicitly supported by the material.
If no valid calculation is possible, clearly report that
instead of inventing unsupported facts.

Every question must include:
type, question, topic, explanation, page, difficulty.

For MCQs, include:
options (exactly four), correct_answer.

For short-answer questions, include:
reference_answer.

For numerical questions, include:
numeric_answer (a number), tolerance (a non-negative number),
and unit (use an empty string if no unit applies).

Use types exactly: mcq, short_answer, numerical.
Use source page numbers from the supplied material.
Difficulty must be Easy, Medium, or Hard.

Return only valid JSON:
{
  "questions": [
    {
      "type": "mcq",
      "question": "Question text",
      "topic": "Topic",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "Exact correct option",
      "explanation": "Source-supported explanation",
      "page": 1,
      "difficulty": "Easy"
    },
    {
      "type": "short_answer",
      "question": "Question text",
      "topic": "Topic",
      "reference_answer": "Expected answer",
      "explanation": "Why this answer is expected",
      "page": 1,
      "difficulty": "Medium"
    },
    {
      "type": "numerical",
      "question": "Calculation question with all needed values",
      "topic": "Topic",
      "numeric_answer": 12.5,
      "tolerance": 0.1,
      "unit": "",
      "explanation": "Steps and result supported by the material",
      "page": 1,
      "difficulty": "Medium"
    }
  ]
}
"""

    user_prompt = (
        f"Create exactly {number_of_questions} questions: "
        "3 MCQs, 1 short answer, and 1 numerical question.\n"
        f"Focus topic: {focus_topic or 'All topics'}.\n"
        "Keep every question within the evidence in the material.\n\n"
        f"Study material:\n{context}"
    )

    data = call_groq_json(system_prompt, user_prompt)
    questions = data.get("questions", [])

    if len(questions) != number_of_questions:
        raise ValueError(
            f"Expected {number_of_questions} questions, "
            f"but received {len(questions)}. "
            f"Received question types: "
            f"{[q.get('type', 'missing') for q in questions]}"
        )
    types = [q.get("type") for q in questions]
    if types.count("mcq") != 3:
        raise ValueError("The quiz must contain exactly 3 MCQs.")
    if types.count("short_answer") != 1:
        raise ValueError("The quiz must contain 1 short-answer question.")
    if types.count("numerical") != 1:
        raise ValueError("The quiz must contain 1 numerical question.")

    for q in questions:
        if not all(q.get(k) for k in ["question", "topic", "explanation"]):
            raise ValueError("A question is missing required information.")

        if not isinstance(q.get("page"), int):
            raise ValueError("A question is missing its source page.")

        if q["type"] == "mcq":
            if (
                len(q.get("options", [])) != 4
                or q.get("correct_answer") not in q.get("options", [])
            ):
                raise ValueError("An MCQ has invalid options or answer.")

        elif q["type"] == "short_answer":
            if not q.get("reference_answer"):
                raise ValueError("A short-answer question has no reference answer.")

        elif q["type"] == "numerical":
            if not isinstance(q.get("numeric_answer"), (int, float)):
                raise ValueError("The numerical answer is not a number.")
            if not isinstance(q.get("tolerance"), (int, float)):
                raise ValueError("The numerical tolerance is invalid.")
            if q["tolerance"] < 0:
                raise ValueError("The numerical tolerance cannot be negative.")

    return questions

def grade_short_answer(question, student_answer, reference_answer, source_text):
    system_prompt = """
You grade a student's short answer using only the supplied study source
and reference answer.

Accept equivalent wording if the key meaning is correct.
Do not require exact wording.
Do not give credit for claims contradicted by the source.
If the response is blank or misses a key idea, mark it incorrect.

Return only JSON:
{
  "correct": true,
  "feedback": "Brief explanation of the judgement"
}
"""

    user_prompt = f"""
Question:
{question}

Reference answer:
{reference_answer}

Student answer:
{student_answer}

Study source:
{source_text}

Decide whether the student's answer captures the essential meaning.
"""

    result = call_groq_json(system_prompt, user_prompt)

    if not isinstance(result.get("correct"), bool):
        raise ValueError("The short-answer grader returned an invalid result.")

    return result

def verify_evidence(question, retrieved_chunks):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    context = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context += (
            f"\nPassage {i} | Page {chunk['page']}\n"
            f"{chunk['text']}\n"
        )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """
You are an evidence checker for a study tutor.

Decide whether the supplied passages contain enough
information to answer the user's exact question.

Return SUPPORTED only if the necessary facts are
explicitly stated or can be directly inferred from
the passages.

Return INSUFFICIENT if important facts are missing,
the passages are merely related to the topic, or
answering would require outside knowledge.

Do not use your own general knowledge.

Your response must start with exactly one of:
SUPPORTED
INSUFFICIENT

Then give a brief explanation.
"""
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Study passages:\n{context}"
                )
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def generate_answer(question, retrieved_chunks):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    context = ""

    for chunk in retrieved_chunks:
        context += (
            f"\n[Page {chunk['page']}]\n"
            f"{chunk['text']}\n"
        )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """
You are a helpful study tutor.

Answer using only the supplied study passages.
Explain concepts clearly and in simple language.
Cite supporting pages using [Page N].

Do not invent facts or claim that a page supports
something unless that page contains the evidence.
If the passages do not answer part of the question,
state that limitation clearly.
"""
            },
            {
                "role": "user",
                "content": (
                    f"Study passages:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Build searchable knowledge base"):

        with st.spinner("Reading PDF and creating embeddings..."):

            chunks = create_chunks(uploaded_file.getvalue())

            if not chunks:
                st.error(
                    "No selectable text found in this PDF."
                )
                st.stop()

            
            with st.spinner("Discovering major topics and subtopics..."):
                topic_hierarchy = discover_topic_hierarchy(chunks)

            with st.spinner("Tagging each content chunk..."):
                chunks = tag_content_chunks(chunks, topic_hierarchy)

            st.session_state["topic_hierarchy"] = topic_hierarchy
            st.session_state["chunks"] = chunks

            st.success(
                f"Topic discovery complete! "
                f"Tagged {len(chunks)} content chunks."
            )


            model = load_embedding_model()

            texts = [chunk["text"] for chunk in chunks]

            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            embeddings = np.asarray(
                embeddings,
                dtype="float32"
            )

            # Inner product on normalized vectors
            # is equivalent to cosine similarity.
            index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

            index.add(embeddings)

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index

        st.success(
            f"Knowledge base ready: {len(chunks)} chunks."
        )

    if "index" in st.session_state:

        question = st.text_input(
            "Ask a question about your notes",
            placeholder="What is the main idea of this document?"
        )

        if st.button("Find relevant passages") and question.strip():

            model = load_embedding_model()

            query_embedding = model.encode(
                [question],
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype("float32")

            index = st.session_state["index"]
            chunks = st.session_state["chunks"]

            k = min(3, len(chunks))
            scores, indices = index.search(query_embedding, k)

            # Keep passages that pass the temporary threshold
            retrieved_chunks = []
            retrieved_scores = []

            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and float(score) >= 0.15:
                    retrieved_chunks.append(chunks[int(idx)])
                    retrieved_scores.append(float(score))

            st.subheader("AI Tutor's Answer")

            if not retrieved_chunks:
                st.warning(
                    "I couldn't find sufficiently relevant passages "
                    "in your uploaded material. Try rephrasing your question."
                )

            else:
                try:
                    with st.spinner("Checking whether the passages support your question..."):
                        evidence_result = verify_evidence(
                            question, retrieved_chunks
                        )

                    # Check the first word of the verifier's response
                    first_word = evidence_result.split()[0].upper().strip(".,:;") if evidence_result.split() else ""

                    if first_word == "SUPPORTED":

                        with st.spinner("Generating your explanation..."):
                            answer = generate_answer(
                                question, retrieved_chunks
                            )

                        st.markdown(answer)

                    elif first_word == "INSUFFICIENT":

                        st.warning(
                            "The uploaded material does not contain "
                            "enough evidence to answer this question reliably."
                        )

                        with st.expander("Why couldn't I answer?"):
                            st.write(evidence_result)

                    else:
                        st.warning(
                            "I couldn't verify the evidence confidently. "
                            "Please try rephrasing your question."
                        )

                except Exception as error:
                    st.error("An error occurred while processing your question.")
                    st.caption(f"Technical details: {error}")

            # Keep retrieved passages available for debugging
            with st.expander("View retrieved passages and scores"):
                if retrieved_chunks:
                    for number, (chunk, score) in enumerate(
                        zip(retrieved_chunks, retrieved_scores),
                        start=1
                    ):
                        st.markdown(
                            f"**Passage {number} — Page {chunk['page']}**"
                        )
                        with st.expander("View retrieved passages and scores"):
                            if retrieved_chunks:
                                for number, (chunk, score) in enumerate(
                                    zip(retrieved_chunks, retrieved_scores),
                                    start=1
                                ):
                                    st.markdown(
                                        f"**Passage {number} — Page {chunk['page']}**"
                                    )

                                    # ADD THIS CODE HERE
                                    st.caption(
                                        f"Major topic: {chunk.get('major_topic', 'Unclassified')} | "
                                        f"Subtopic: {chunk.get('subtopic', 'Unclassified')}"
                                    )

                                    if chunk.get("key_topics"):
                                        st.write(
                                            "Key concepts: " + ", ".join(chunk["key_topics"])
                                        )

                                    # Existing code — keep it
                                    st.caption(f"Cosine similarity: {score:.3f}")
                                    st.write(chunk["text"])
                                    st.divider()
                            else:
                                st.write("No passages passed the retrieval threshold.")
                        st.caption(f"Cosine similarity: {score:.3f}")
                        st.write(chunk["text"])
                        st.divider()
                else:
                    st.write("No passages passed the retrieval threshold.")

st.divider()
st.header("📝 Practice Quiz")
st.write("Test your understanding of the uploaded study material.")

if "quiz_questions" not in st.session_state:
    st.session_state["quiz_questions"] = []

if "quiz_submitted" not in st.session_state:
    st.session_state["quiz_submitted"] = False

quiz_hierarchy = st.session_state.get("topic_hierarchy", [])

quiz_topic_options = ["All topics"] + [
    topic["name"] for topic in quiz_hierarchy
]

selected_quiz_topic = st.selectbox(
    "Choose a topic for your quiz",
    quiz_topic_options,
    key="selected_quiz_topic"
)

if st.button("Generate Balanced Quiz"):
    if "chunks" not in st.session_state:
        st.warning("Upload a PDF and build the knowledge base first.")
    else:
        all_chunks = st.session_state["chunks"]

        if selected_quiz_topic == "All topics":
            quiz_chunks = all_chunks
        else:
            quiz_chunks = [
                chunk for chunk in all_chunks
                if chunk.get("major_topic") == selected_quiz_topic
            ]

        if not quiz_chunks:
            st.warning("No content was found for this topic.")
        else:
            try:
                with st.spinner("Generating your balanced quiz..."):
                    questions = generate_quiz(
                        quiz_chunks[:8],
                        number_of_questions=5,
                        focus_topic=(
                            None if selected_quiz_topic == "All topics"
                            else selected_quiz_topic
                        )
                    )

                st.session_state["quiz_questions"] = questions
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_attempt_recorded"] = False
                st.session_state["quiz_results"] = None
                st.success("Balanced quiz generated!")

            except Exception as error:
                st.error("Could not generate the balanced quiz.")
                st.caption(f"Details: {error}")


questions = st.session_state.get("quiz_questions", [])

if questions:
    with st.form("balanced_quiz_form"):
        for i, q in enumerate(questions):
            st.subheader(f"Question {i + 1}")
            st.write(q["question"])
            st.caption(
                f"Type: {q['type'].replace('_', ' ').title()} | "
                f"Difficulty: {q['difficulty']} | "
                f"Source: Page {q['page']}"
            )

            if q["type"] == "mcq":
                st.radio(
                    "Choose one answer:",
                    q["options"],
                    key=f"quiz_answer_{i}",
                    index=None
                )

            elif q["type"] == "short_answer":
                st.text_area(
                    "Write your answer:",
                    key=f"quiz_answer_{i}"
                )

            elif q["type"] == "numerical":
                st.text_input(
                    f"Enter your numerical answer"
                    + (f" ({q.get('unit', '')})" if q.get("unit") else "")
                    + ":",
                    key=f"quiz_answer_{i}"
                )

        submitted = st.form_submit_button("Submit Quiz")

    if submitted:
        results = []
        score = 0

        with st.spinner("Checking your answers..."):
            for i, q in enumerate(questions):
                answer = st.session_state.get(f"quiz_answer_{i}", "")
                correct = False
                feedback = ""

                if q["type"] == "mcq":
                    correct = answer == q["correct_answer"]
                    feedback = q["explanation"]

                elif q["type"] == "short_answer":
                    # Find source text from the cited page.
                    source_text = "\n".join(
                        chunk["text"]
                        for chunk in st.session_state["chunks"]
                        if chunk["page"] == q["page"]
                    )

                    if answer.strip():
                        try:
                            grading = grade_short_answer(
                                q["question"],
                                answer,
                                q["reference_answer"],
                                source_text
                            )
                            correct = grading["correct"]
                            feedback = grading["feedback"]
                        except Exception:
                            correct = False
                            feedback = (
                                "Automatic grading failed. Compare your answer "
                                "with the reference answer below."
                            )
                    else:
                        feedback = "No answer was entered."

                elif q["type"] == "numerical":
                    try:
                        student_number = float(answer.strip())
                        correct = (
                            abs(student_number - q["numeric_answer"])
                            <= q["tolerance"]
                        )
                    except (ValueError, AttributeError):
                        correct = False

                    feedback = q["explanation"]

                if correct:
                    score += 1

                results.append({
                    "correct": correct,
                    "feedback": feedback,
                    "answer": answer
                })

        st.session_state["quiz_results"] = results
        st.session_state["quiz_score"] = score
        st.session_state["quiz_submitted"] = True

        # Record this attempt once, not on every Streamlit rerun.
        if not st.session_state.get("quiz_attempt_recorded", False):
            learner = st.session_state["learner_model"]

            for q, result in zip(questions, results):
                topic = q["topic"]

                if topic not in learner:
                    learner[topic] = {"attempts": 0, "correct": 0}

                learner[topic]["attempts"] += 1
                if result["correct"]:
                    learner[topic]["correct"] += 1

            st.session_state["learner_model"] = learner
            st.session_state["quiz_attempt_recorded"] = True

    if st.session_state.get("quiz_submitted"):
        results = st.session_state.get("quiz_results") or []

        st.subheader("Quiz Results")

        for i, (q, result) in enumerate(zip(questions, results)):
            if result["correct"]:
                st.success(f"Question {i + 1}: Correct")
            else:
                st.error(f"Question {i + 1}: Needs review")

            st.write("Feedback:", result["feedback"])

            if q["type"] == "mcq":
                st.write("Correct answer:", q["correct_answer"])
            elif q["type"] == "short_answer":
                st.write("Reference answer:", q["reference_answer"])
            elif q["type"] == "numerical":
                expected = q["numeric_answer"]
                unit = q.get("unit", "")
                st.write(f"Expected answer: {expected} {unit}".strip())

            st.write("Explanation:", q["explanation"])
            st.caption(f"Review source: Page {q['page']}")
            st.divider()

        st.metric("Your Score", f"{st.session_state['quiz_score']} / {len(questions)}")

st.divider()
st.header("📊 Your Learning Progress")

learner = st.session_state["learner_model"]

if not learner:
    st.info("Complete a quiz to start tracking your progress.")

else:
    for topic, data in learner.items():

        attempts = data["attempts"]
        correct = data["correct"]

        mastery = ((correct + 1) / (attempts + 2)) * 100

        st.subheader(topic)

        st.progress(mastery / 100)

        st.write(
            f"Mastery estimate: {mastery:.0f}% "
            f"({correct} correct out of {attempts})"
        )

    weak_topics = [
        topic
        for topic, data in learner.items()
        if (data["correct"] + 1) / (data["attempts"] + 2) < 0.6
    ]

    if weak_topics:
        st.warning(
            "Topics to revise: " + ", ".join(weak_topics)
        )
    else:
        st.success(
            "Good progress! No tracked topic is currently "
            "below 60% mastery."
        )
    

st.divider()
st.header("🗂️ Study Material Topic Map")

topic_hierarchy = st.session_state.get("topic_hierarchy", [])

if not topic_hierarchy:
    st.info("Build the knowledge base to discover topics.")

else:
    for major in topic_hierarchy:
        with st.expander("📘 " + major["name"]):
            for subtopic in major.get("subtopics", []):
                st.markdown("### " + subtopic["name"])

                key_topics = subtopic.get("key_topics", [])

                if key_topics:
                    st.write("Key concepts:", ", ".join(key_topics))

                matching_chunks = [
                    chunk
                    for chunk in st.session_state.get("chunks", [])
                    if chunk.get("major_topic") == major["name"]
                    and chunk.get("subtopic") == subtopic["name"]
                ]

                st.caption(
                    f"{len(matching_chunks)} tagged content unit(s)"
                )

                for chunk in matching_chunks:
                    st.markdown(f"**Source: Page {chunk['page']}**")
                    st.write(chunk["text"])
                    st.divider()
