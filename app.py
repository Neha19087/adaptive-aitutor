import json
import streamlit as st
import pymupdf
import numpy as np
import faiss
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import re
from urllib.parse import urlparse, parse_qs

st.set_page_config(
    page_title="Adaptive AI Tutor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Main page */
    .stApp {
        background-color: #F6F8FC;
        color: #1E293B;
    }

    /* Default text */
    .stApp,
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp li,
    .stApp div {
        color: #1E293B;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #172554 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #172554;
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
    }

    [data-testid="stMetric"] label {
        color: #475569 !important;
    }

    [data-testid="stMetricValue"] {
        color: #172554 !important;
    }

    /* Inputs */
    .stTextInput input,
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1;
    }

    /* Select boxes */
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #FFFFFF;
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #4F46E5;
        color: #FFFFFF !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #4338CA;
        color: #FFFFFF !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
    }

    /* Content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    
    /* Fix PDF upload box */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 14px !important;
        padding: 16px !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 1px dashed #94A3B8 !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section * {
        color: #1E293B !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] small {
        color: #64748B !important;
    }
    
.dashboard-hero {
    background: linear-gradient(120deg, #172554, #4338CA);
    padding: 32px;
    border-radius: 18px;
    margin-bottom: 24px;
}

.dashboard-hero .hero-label {
    color: #C7D2FE !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.dashboard-hero h1 {
    color: #FFFFFF !important;
    font-size: 30px;
}

.dashboard-hero p {
    color: #E0E7FF !important;
    line-height: 1.7;
}


</style>
""", unsafe_allow_html=True)


def get_weak_topics(learner_model):
    weak_topics = []

    for topic, data in learner_model.items():
        attempts = data["attempts"]
        correct = data["correct"]

        mastery = (correct + 1) / (attempts + 2)

        if mastery < 0.6:
            weak_topics.append(topic)

    return weak_topics


def get_revision_advice(attempts, correct, mastery):
    accuracy = correct / attempts if attempts > 0 else 0

    # A few attempts are not enough for a confident estimate
    if attempts < 3:
        return (
            "Your progress estimate is still based on a small number "
            "of questions. Review the material and try a few more "
            "questions to get a clearer picture of your understanding."
        )

    # Very low estimated mastery
    if mastery < 40:
        return (
            "Prioritize this topic. Revisit the basic concepts and "
            "examples in your notes before attempting another quiz."
        )

    # Below the current 60% threshold
    if mastery < 60:
        return (
            "Review the key definitions and supporting examples, "
            "then attempt a targeted quiz to strengthen your understanding."
        )

    # Maintenance advice for topics outside the weak group
    return (
        "Continue practising this topic periodically to reinforce "
        "your understanding."
    )

# Initialize the learner model
if "learner_model" not in st.session_state:
    st.session_state["learner_model"] = {}




# Dashboard header
st.markdown("""
# 📚 Adaptive AI Tutor

### Your personal AI-powered learning workspace

Upload your notes, understand difficult concepts,
practise with quizzes, and track your learning progress.
""")

st.divider()

# Get existing application data
learner = st.session_state["learner_model"]
chunks = st.session_state.get("chunks", [])
hierarchy = st.session_state.get("topic_hierarchy", [])

total_questions = sum(
    data["attempts"] for data in learner.values()
)

total_correct = sum(
    data["correct"] for data in learner.values()
)

accuracy = (
    100 * total_correct / total_questions
    if total_questions > 0 else 0
)

weak_count = len(get_weak_topics(learner))

# Dashboard metric cards
st.subheader("Your Learning Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📄 Study Chunks", len(chunks))

with col2:
    st.metric("📚 Major Topics", len(hierarchy))

with col3:
    st.metric("🎯 Questions Attempted", total_questions)

with col4:
    st.metric("📈 Accuracy", f"{accuracy:.0f}%")

st.caption(
    f"You currently have {weak_count} topic(s) "
    "that may need additional practice."
)

st.divider()


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_chunks_for_topic(topic, chunks, top_k=3):
    if not chunks:
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [f"Study material explaining {topic}"],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    index = st.session_state.get("index")

    if index is None:
        return []

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for idx in indices[0]:
        if idx >= 0:
            results.append(chunks[int(idx)])

    return results

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


def extract_youtube_id(url):
    """Extract a YouTube video ID from a supported URL."""
    parsed = urlparse(url.strip())

    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.strip("/").split("/")[0]

    if parsed.hostname and parsed.hostname.endswith("youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]

        if parsed.path.startswith(("/embed/", "/shorts/")):
            return parsed.path.split("/")[2]

    return None


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{remaining:02d}"


def get_source_label(chunk):
    source_type = chunk.get("source_type", "pdf")
    source_name = chunk.get("source_name", "")

    if source_type in ("video", "youtube"):
        timestamp = chunk.get("timestamp")

        if timestamp:
            return f"Video {source_name} at {timestamp}"

        return f"Video {source_name}"

    page = chunk.get("page")

    if page is not None:
        return f"Page {page}"

    return "Unknown source"

def transcribe_uploaded_video(video_file, video_name):
    """Transcribe an uploaded video using Groq Whisper."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")

    video_bytes = video_file.getvalue()

    # Groq's transcription API has a file-size limit.
    if len(video_bytes) > 25 * 1024 * 1024:
        raise ValueError(
            "This video exceeds the 25 MB upload limit for this "
            "initial version. Try a smaller video."
        )

    client = Groq(api_key=api_key)

    result = client.audio.transcriptions.create(
        file=(video_name, video_bytes),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )

    segments = getattr(result, "segments", None)

    if not segments:
        raise ValueError(
            "No timestamped speech was returned. "
            "Check that the video contains audible speech."
        )

    chunks = []

    for segment in segments:
        text = getattr(segment, "text", "").strip()

        if not text:
            continue

        start = float(getattr(segment, "start", 0))
        end = float(getattr(segment, "end", start))

        chunks.append({
            "text": text,
            "page": None,
            "source_type": "video",
            "source_name": video_name,
            "timestamp_start": start,
            "timestamp_end": end,
            "timestamp": format_timestamp(start),
        })

    if not chunks:
        raise ValueError("The video transcript is empty.")

    return chunks



def get_youtube_transcript(url):
    video_id = extract_youtube_id(url)

    if not video_id:
        raise ValueError(
            "Invalid YouTube URL. Please paste a normal video link."
        )

    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()

    # Try English captions first, then other available languages.
    try:
        transcript = api.fetch(
            video_id,
            languages=["en"]
        )
    except Exception as english_error:
        try:
            transcript_list = api.list(video_id)

            available = list(transcript_list)

            if not available:
                raise ValueError(
                    "No captions are available for this video."
                )

            transcript = available[0].fetch()

        except Exception as fallback_error:
            raise ValueError(
                "Could not fetch captions. The video may have "
                "no accessible transcript, or YouTube may be "
                f"blocking the request. Details: {fallback_error}"
            ) from english_error

    chunks = []

    for item in transcript:
        text = item.text.strip()

        if not text:
            continue

        start = float(item.start)
        duration = float(item.duration)

        chunks.append({
            "text": text,
            "page": None,
            "source_type": "youtube",
            "source_name": url,
            "timestamp_start": start,
            "timestamp_end": start + duration,
            "timestamp": format_timestamp(start)
        })

    if not chunks:
        raise ValueError(
            "The transcript was fetched, but it contained no usable text."
        )

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
    """Generate concept-focused MCQs with verified source locations."""

    if not chunks:
        raise ValueError("No study material is available for the quiz.")

    # Sample across the material instead of using only its beginning.
    max_chunks = 20

    if len(chunks) > max_chunks:
        positions = np.linspace(
            0, len(chunks) - 1, max_chunks, dtype=int
        )
        selected_chunks = [chunks[int(i)] for i in positions]
    else:
        selected_chunks = chunks

    # Give each passage an ID. The model must cite this ID;
    # Python will look up the real page or timestamp.
    context_parts = []

    for i, chunk in enumerate(selected_chunks):
        context_parts.append(
            f"\n[CHUNK_ID: {i}]\n"
            f"[SOURCE: {get_source_label(chunk)}]\n"
            f"[CONTENT:]\n{chunk['text']}\n"
        )

    context = "\n".join(context_parts)

    system_prompt = """
You are an expert university-level question setter.

Create challenging, useful questions that test understanding
of the actual subject matter in the supplied study passages.

STRICT CONTENT RULES:
- Use only the supplied study material.
- Focus on important concepts, principles, definitions,
  mechanisms, applications, comparisons, and cause-and-effect.
- Prefer questions that require understanding or application,
  not merely remembering isolated words.
- Do not ask about the introduction, conclusion, speaker,
  author, narration, video structure, or when something was said.
- Never ask when or where a concept was mentioned.
- Never ask about timestamps.
- Avoid generic questions that could apply to any video.
- Do not repeat the same concept in multiple questions.
- Do not invent facts or use outside knowledge.
- If the material does not support a meaningful question,
  choose a different concept that is supported.

MCQ RULES:
- Return exactly 5 MCQs.
- Each question must have exactly 4 options.
- Exactly one option must be correct.
- Incorrect options should be plausible but contradicted by,
  or inconsistent with, the supplied material.
- Explain why the correct answer is supported by the material.
- Use Easy, Medium, or Hard for difficulty.
- Each question must have a topic.
- Each question must cite one CHUNK_ID from the passages.
- Choose the chunk that directly supports the answer.
- Never invent a CHUNK_ID.

Return ONLY valid JSON in this structure:
{
  "questions": [
    {
      "type": "mcq",
      "question": "A meaningful subject-focused question",
      "topic": "Specific concept",
      "options": [
        "First option",
        "Second option",
        "Third option",
        "Fourth option"
      ],
      "correct_answer": "Exact text of the correct option",
      "explanation": "Why the correct answer follows from the source",
      "source_chunk_id": 0,
      "difficulty": "Medium"
    }
  ]
}
"""

    user_prompt = (
        f"Generate exactly {number_of_questions} MCQs.\n"
        f"Focus topic: {focus_topic or 'All topics'}.\n"
        "Use the most important concepts in the supplied material. "
        "Do not make questions about presentation or narration. "
        "Use source_chunk_id to identify the passage supporting "
        "each question.\n\n"
        f"Study passages:\n{context}"
    )

    data = call_groq_json(system_prompt, user_prompt)
    questions = data.get("questions", [])

    if len(questions) != number_of_questions:
        raise ValueError(
            f"Expected {number_of_questions} questions, "
            f"but received {len(questions)}."
        )

    for q in questions:
        if q.get("type") != "mcq":
            raise ValueError("The quiz must contain only MCQs.")

        if not all(
            q.get(field)
            for field in ["question", "topic", "explanation"]
        ):
            raise ValueError("A question is missing required information.")

        options = q.get("options", [])

        if (
            not isinstance(options, list)
            or len(options) != 4
            or not all(isinstance(option, str) for option in options)
            or len(set(options)) != 4
            or q.get("correct_answer") not in options
        ):
            raise ValueError("An MCQ has invalid options or answer.")

        source_id = q.get("source_chunk_id")

        if (
            not isinstance(source_id, int)
            or isinstance(source_id, bool)
            or source_id < 0
            or source_id >= len(selected_chunks)
        ):
            raise ValueError("A question has an invalid source passage.")

        # Assign source metadata from the real chunk, not from the AI.
        source_chunk = selected_chunks[source_id]

        q["page"] = source_chunk.get("page")
        q["timestamp"] = source_chunk.get("timestamp")
        q["source_type"] = source_chunk.get("source_type", "pdf")
        q["source_name"] = source_chunk.get("source_name", "")

        q["difficulty"] = q.get("difficulty", "Medium")

        if q["difficulty"] not in ("Easy", "Medium", "Hard"):
            q["difficulty"] = "Medium"

        if q["page"] is None and not q["timestamp"]:
            raise ValueError(
                "The selected source passage has no page or timestamp."
            )

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
        source = get_source_label(chunk)

        context += (
            f"\nPassage {i} | Source: {source}\n"
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
        source = get_source_label(chunk)

        context += (
            f"\n[Source: {source}]\n"
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
Cite each claim using the exact source label provided in the passages.
For PDF sources, use [Page N].
For video and YouTube sources, use [Video HH:MM:SS].
Never write [Page None] or invent a source citation.

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

# SIDEBAR NAVIGATION
st.sidebar.title("📚 Adaptive AI Tutor")
st.sidebar.caption("Your personal learning workspace")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Study Material & AI Tutor",
        "Practice Quiz",
        "My Progress",
        "Topic Map",
    ],
    key="main_navigation",
)

st.sidebar.divider()
st.sidebar.caption("Learn • Practise • Improve")


# DASHBOARD PAGE
if page == "Dashboard":
    st.markdown("""
    <div class="dashboard-hero">
        <div class="hero-label">YOUR PERSONAL LEARNING SPACE</div>
        <h1>Welcome to your learning dashboard 👋</h1>
        <p>
            Practise smarter, track your progress, and focus
            on the topics that need your attention.
        </p>
    </div>
    """, unsafe_allow_html=True)

    learner = st.session_state["learner_model"]
    chunks = st.session_state.get("chunks", [])
    hierarchy = st.session_state.get("topic_hierarchy", [])

    attempts = sum(x["attempts"] for x in learner.values())
    correct = sum(x["correct"] for x in learner.values())
    incorrect = attempts - correct

    accuracy = 100 * correct / attempts if attempts else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Study Chunks", len(chunks))
    col2.metric("Major Topics", len(hierarchy))
    col3.metric("Questions Attempted", attempts)
    col4.metric("Quiz Accuracy", f"{accuracy:.1f}%")

    st.subheader("Quiz Performance")
    if attempts:
        st.bar_chart({
            "Correct": [correct],
            "Incorrect": [incorrect],
        })
    else:
        st.info("Complete a quiz to see your performance.")

    st.subheader("Topic Mastery")
    if learner:
        mastery = {
            topic: round(
                (data["correct"] + 1) /
                (data["attempts"] + 2) * 100,
                1,
            )
            for topic, data in learner.items()
        }
        st.bar_chart(mastery)
    else:
        st.info("Complete a quiz to see topic mastery.")

    weak_topics = get_weak_topics(learner)
    st.subheader("Topics to Revise")
    if weak_topics:
        for topic in weak_topics:
            st.warning(topic)
    else:
        st.info("No weak topics are currently recorded.")


if page == "Study Material & AI Tutor":
    st.subheader("📂 Your Study Material")
    st.write(
        "Start by uploading a PDF or video link containing your notes "
        "or course material."
    )

    
    st.subheader("📚 Add Your Learning Material")

    uploaded_file = st.file_uploader(
        "Upload PDF notes",
        type=["pdf"],
        key="pdf_upload",
    )

    uploaded_videos = st.file_uploader(
        "Upload video lectures",
        type=["mp4", "webm", "mpeg", "mpg"],
        accept_multiple_files=True,
        key="video_upload",
    )

    youtube_url = st.text_input(
        "Or paste a YouTube lecture URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="youtube_url",
    )

    st.caption(
        "You can combine PDF notes, uploaded videos, "
        "and a YouTube lecture in one knowledge base."
    )



    
    if st.button("Build searchable knowledge base"):

        all_chunks = []
        source_errors = []

        # 1. Read PDF notes, if provided
        if uploaded_file is not None:
            with st.spinner("Reading PDF notes..."):
                try:
                    pdf_chunks = create_chunks(
                        uploaded_file.getvalue()
                    )

                    for chunk in pdf_chunks:
                        chunk["source_type"] = "pdf"
                        chunk["source_name"] = uploaded_file.name
                        chunk["timestamp"] = None

                    all_chunks.extend(pdf_chunks)

                    if not pdf_chunks:
                        source_errors.append(
                            f"No selectable text found in {uploaded_file.name}"
                        )

                except Exception as e:
                    source_errors.append(
                        f"Could not read PDF: {e}"
                    )

        # 2. Transcribe uploaded video lectures
        for video_file in uploaded_videos:
            with st.spinner(
                f"Transcribing {video_file.name}..."
            ):
                try:
                    video_chunks = transcribe_uploaded_video(
                        video_file,
                        video_file.name
                    )
                    all_chunks.extend(video_chunks)

                except Exception as e:
                    source_errors.append(
                        f"Could not process {video_file.name}: {e}"
                    )

        # 3. Fetch a YouTube transcript, if a URL was provided
        if youtube_url.strip():
            with st.spinner("Fetching YouTube transcript..."):
                try:
                    youtube_chunks = get_youtube_transcript(
                        youtube_url.strip()
                    )
                    all_chunks.extend(youtube_chunks)

                except Exception as e:
                    source_errors.append(
                        f"Could not process YouTube URL: {e}"
                    )

        # Display source errors without hiding successful sources
        for error in source_errors:
            st.warning(error)

        # Stop if no source produced usable text
        if not all_chunks:
            st.error(
                "No usable learning material was found. "
                "Upload a readable PDF or video, or provide "
                "a YouTube URL with available captions."
            )
            st.stop()

        # 4. Discover topics across ALL uploaded sources
        with st.spinner(
            "Discovering major topics and subtopics..."
        ):
            topic_hierarchy = discover_topic_hierarchy(
                all_chunks
            )

        # 5. Tag every PDF and video transcript chunk
        with st.spinner("Tagging each content chunk..."):
            all_chunks = tag_content_chunks(
                all_chunks,
                topic_hierarchy
            )

        st.session_state["topic_hierarchy"] = topic_hierarchy
        st.session_state["chunks"] = all_chunks

        # 6. Create embeddings
        with st.spinner(
            "Creating embeddings and searchable index..."
        ):
            model = load_embedding_model()

            texts = [
                chunk["text"] for chunk in all_chunks
            ]

            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            embeddings = np.asarray(
                embeddings,
                dtype="float32"
            )

            # Normalized vectors + inner product = cosine similarity
            index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

            index.add(embeddings)

            st.session_state["index"] = index

        st.success(
            f"Knowledge base ready! Indexed "
            f"{len(all_chunks)} text chunks."
        )

        st.write("Sources successfully added:")

        pdf_count = sum(
            1 for c in all_chunks
            if c.get("source_type") == "pdf"
        )

        video_count = sum(
            1 for c in all_chunks
            if c.get("source_type") == "video"
        )

        youtube_count = sum(
            1 for c in all_chunks
            if c.get("source_type") == "youtube"
        )

        st.write(f"- PDF chunks: {pdf_count}")
        st.write(f"- Uploaded video chunks: {video_count}")
        st.write(f"- YouTube transcript chunks: {youtube_count}")


    if "index" in st.session_state and "chunks" in st.session_state:
        chunks = st.session_state["chunks"]
        index = st.session_state["index"]

        st.write(
            f"Knowledge base ready: {len(chunks)} chunks."
        )     

        question = st.text_input(
            "Ask a question about your notes",
            placeholder="What is the main idea of this material?"
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
if page == "Practice Quiz":
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
                            quiz_chunks,
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
                source_label = (
                    f"Page {q['page']}"
                    if q.get("page") is not None
                    else f"Video {q.get('timestamp', 'timestamp unavailable')}"
                )

                st.caption(
                    f"Type: {q['type'].replace('_', ' ').title()} | "
                    f"Difficulty: {q['difficulty']} | "
                    f"Source: {source_label}"
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
                
                source_label = (
                    f"Page {q['page']}"
                    if q.get("page") is not None
                    else f"Video at {q.get('timestamp', 'timestamp unavailable')}"
                )
                st.caption(f"Review source: {source_label}")

                st.divider()

            st.metric("Your Score", f"{st.session_state['quiz_score']} / {len(questions)}")
if page == "My Progress":
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
    st.header("🎯 Practice Weak Topics")

    learner = st.session_state["learner_model"]
    weak_topics = get_weak_topics(learner)

    if not learner:
        st.info(
            "Complete a quiz first to identify topics "
            "that need more practice."
        )

    elif not weak_topics:
        st.success(
            "No tracked topics are currently below "
            "the 60% mastery threshold."
        )
        st.write(
            "Keep practising to reinforce your understanding."
        )

    else:
        # Rank topics from lowest to highest estimated mastery.
        weak_topics = sorted(
            weak_topics,
            key=lambda topic: (
                learner[topic]["correct"] + 1
            ) / (
                learner[topic]["attempts"] + 2
            )
        )

        selected_topic = st.selectbox(
            "Choose a topic to revise",
            weak_topics,
            key="weak_topic_selector"
        )

        data = learner[selected_topic]
        attempts = data["attempts"]
        correct = data["correct"]

        mastery = (
            (correct + 1) / (attempts + 2)
        ) * 100

        st.metric("Estimated mastery", f"{mastery:.0f}%")
        st.progress(mastery / 100)

        st.write(
            f"Quiz performance: {correct} correct "
            f"out of {attempts} attempts."
        )

        st.info(
            get_revision_advice(attempts, correct, mastery)
        )

        if st.button("Show Revision Material"):
            with st.spinner("Finding relevant passages..."):
                passages = retrieve_chunks_for_topic(
                    selected_topic,
                    st.session_state.get("chunks", []),
                    top_k=3
                )

            if passages:
                st.subheader("Recommended passages to review")

                for passage in passages:
                    st.caption(f"Source: Page {passage['page']}")
                    st.write(passage["text"])
                    st.divider()
            else:
                st.warning(
                    "Build the knowledge base first, "
                    "or no relevant passages were found."
                )

        if st.button("Generate Targeted Quiz"):
            chunks = st.session_state.get("chunks", [])

            if not chunks:
                st.warning(
                    "Upload a PDF and build the knowledge base first."
                )
            else:
                try:
                    with st.spinner(
                        f"Creating a quiz about {selected_topic}..."
                    ):
                        relevant_chunks = retrieve_chunks_for_topic(
                            selected_topic,
                            chunks,
                            top_k=8
                        )

                        if not relevant_chunks:
                            st.warning(
                                "No relevant passages were found."
                            )
                        else:
                            targeted_questions = generate_quiz(
                                relevant_chunks,
                                number_of_questions=5,
                                focus_topic=selected_topic
                            )

                            st.session_state["quiz_questions"] = (
                                targeted_questions
                            )
                            st.session_state["quiz_submitted"] = False
                            st.session_state["quiz_attempt_recorded"] = False
                            st.session_state["quiz_results"] = None

                            st.success("Targeted quiz generated!")
                            st.rerun()

                except Exception as error:
                    st.error("Could not generate the targeted quiz.")
                    st.caption(f"Details: {error}")

if page == "Topic Map":
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
