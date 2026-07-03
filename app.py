import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Import your existing pipeline functions
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="AI Video Assistant", page_icon="🎥", layout="wide")

# --- Constants ---
ALLOWED_TYPES = ["mp4", "mov", "avi", "mkv", "mp3", "wav", "m4a"]

# --- Session State (Memory) ---
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def reset_app():
    st.session_state.pipeline_result = None
    st.session_state.chat_history = []
    st.session_state.pending_question = None


def save_uploaded_file(uploaded_file) -> str:
    """Persist an uploaded file to a temp path and return that path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def run_pipeline(source, language_input):
    """Run the full pipeline and store results in session state."""
    reset_app()

    progress = st.progress(0, text="Starting...")
    status_box = st.status("🤖 Running AI Pipeline...", expanded=True)

    try:
        with status_box:
            st.write("📥 Downloading and chunking audio...")
            chunks = process_input(source)
            progress.progress(15, text="Audio ready")

            st.write("🗣️ Transcribing audio (this takes the longest)...")
            transcript = transcribe_all(chunks)
            progress.progress(50, text="Transcript ready")

            st.write("📝 Generating title & summary...")
            title = generate_title(transcript)
            summary = summarize(transcript)
            progress.progress(65, text="Summary ready")

            st.write("🎯 Extracting insights (Action Items, Decisions, Questions)...")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            progress.progress(85, text="Insights extracted")

            st.write("🧠 Building Vector Store for RAG Chat...")
            rag_chain = build_rag_chain(transcript)
            progress.progress(100, text="Done")

            st.session_state.pipeline_result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
                "language": language_input,
            }
            status_box.update(label="✅ Pipeline Complete!", state="complete", expanded=False)

    except Exception as e:
        status_box.update(label="❌ Error occurred!", state="error", expanded=True)
        st.error(f"An error occurred: {e}")
    finally:
        progress.empty()


# --- Sidebar (Inputs) ---
st.sidebar.title("⚙️ Configuration")

source_mode = st.sidebar.radio(
    "Choose input method:",
    ["🔗 YouTube / URL", "📁 Upload File"],
    horizontal=False,
)

source_input = None
uploaded_file = None

if source_mode == "🔗 YouTube / URL":
    source_input = st.sidebar.text_input("Enter YouTube URL or local file path:")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video or audio file",
        type=ALLOWED_TYPES,
        help=f"Supported formats: {', '.join(ALLOWED_TYPES)}",
    )
    if uploaded_file is not None:
        st.sidebar.video(uploaded_file) if uploaded_file.type.startswith("video") else st.sidebar.audio(uploaded_file)
        st.sidebar.caption(f"📦 {uploaded_file.name} — {uploaded_file.size / (1024*1024):.2f} MB")

language_input = st.sidebar.selectbox("Language:", ["english", "hinglish"])

col_a, col_b = st.sidebar.columns(2)
process_btn = col_a.button("🚀 Process", use_container_width=True)
reset_btn = col_b.button("🔄 Reset", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with LangChain, Mistral & Whisper 🚀")

if reset_btn:
    reset_app()
    st.rerun()

# --- Trigger Pipeline ---
if process_btn:
    if source_mode == "🔗 YouTube / URL" and source_input:
        run_pipeline(source_input, language_input)
    elif source_mode == "📁 Upload File" and uploaded_file is not None:
        with st.spinner("Saving uploaded file..."):
            temp_path = save_uploaded_file(uploaded_file)
        run_pipeline(temp_path, language_input)
    else:
        st.sidebar.warning("⚠️ Please provide a URL or upload a file first.")

# --- Dashboard Display ---
if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result

    st.title(f"🎥 {result['title']}")

    # Quick stats row
    word_count = len(result["transcript"].split())
    est_minutes = max(1, word_count // 130)
    m1, m2, m3 = st.columns(3)
    m1.metric("📝 Words Transcribed", f"{word_count:,}")
    m2.metric("⏱️ Est. Video Length", f"~{est_minutes} min")
    m3.metric("🌐 Language", result["language"].capitalize())

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "📝 Transcript"
    ])

    with tab1:
        st.markdown(result["summary"])
        st.download_button("⬇️ Download Summary", result["summary"], file_name="summary.txt")
    with tab2:
        st.markdown(result["action_items"])
        st.download_button("⬇️ Download Action Items", result["action_items"], file_name="action_items.txt")
    with tab3:
        st.markdown(result["key_decisions"])
        st.download_button("⬇️ Download Decisions", result["key_decisions"], file_name="key_decisions.txt")
    with tab4:
        st.markdown(result["open_questions"])
        st.download_button("⬇️ Download Questions", result["open_questions"], file_name="open_questions.txt")
    with tab5:
        with st.expander("Show full raw transcript", expanded=False):
            st.write(result["transcript"])
        st.download_button("⬇️ Download Transcript", result["transcript"], file_name="transcript.txt")

    st.divider()

    # --- Interactive RAG Chat Interface ---
    st.subheader("💬 Chat with this Video")

    # Suggested quick-start questions
    suggestions = [
        "Summarize this in 3 bullet points",
        "What are the main action items?",
        "Were there any disagreements or open questions?",
    ]
    st.caption("Try asking:")
    s_cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        if s_cols[i].button(s, key=f"suggest_{i}", use_container_width=True):
            st.session_state.pending_question = s

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about the video...")
    if st.session_state.pending_question:
        user_query = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(result["rag_chain"], user_query)
                st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.title("Welcome to your AI Video Assistant 🎬")
    st.info("👈 Choose a YouTube URL or upload a video/audio file in the sidebar, then click 'Process' to begin.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 🔗\n**Paste a link**\nYouTube URL or local path")
    c2.markdown("### 📁\n**Upload a file**\nmp4, mov, mp3, wav & more")
    c3.markdown("### 💬\n**Chat instantly**\nAsk questions once processed")