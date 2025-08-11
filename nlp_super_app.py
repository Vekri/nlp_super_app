import streamlit as st
from transformers import pipeline
from langdetect import detect

# --------------------
# Page Setup
# --------------------
st.set_page_config(page_title="🧠 All-in-One NLP App", layout="wide")
st.title("🧠 Natural Language Processing Toolkit")

# --------------------
# Cache Pipelines
# --------------------
@st.cache_resource
def get_pipeline(task, model=None):
    """Load and cache a transformers pipeline."""
    return pipeline(task, model=model) if model else pipeline(task)

# --------------------
# Task Selector
# --------------------
tasks = [
    "Sentiment Analysis",
    "Text Summarization",
    "Named Entity Recognition (NER)",
    "Translation (Multilingual)",
    "Question Answering",
    "Grammar Correction",
    "Text Classification",
    "Language Detection",
    "Keyword Extraction",
    "Chat with a Document (RAG-based)"
]
task = st.selectbox("Choose an NLP Task", tasks)

# --------------------
# Input Handling
# --------------------
if task not in ["Question Answering", "Chat with a Document (RAG-based)"]:
    text = st.text_area("Enter your text", height=200)

# --------------------
# Task Logic
# --------------------
if task == "Sentiment Analysis":
    if st.button("Analyze Sentiment"):
        if text.strip():
            analyzer = get_pipeline("sentiment-analysis")
            result = analyzer(text)[0]
            st.success(f"**Label**: {result['label']}, **Score**: {round(result['score'], 2)}")
        else:
            st.warning("Please enter text to analyze.")

elif task == "Text Summarization":
    if st.button("Summarize"):
        if text.strip():
            summarizer = get_pipeline("summarization")
            summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            st.write("**Summary:**")
            st.info(summary)
        else:
            st.warning("Please enter text to summarize.")

elif task == "Named Entity Recognition (NER)":
    if st.button("Extract Entities"):
        if text.strip():
            ner = get_pipeline("ner", grouped_entities=True)
            entities = ner(text)
            for ent in entities:
                st.write(f"**{ent['entity_group']}**: {ent['word']} ({round(ent['score'], 2)})")
        else:
            st.warning("Please enter text for entity extraction.")

elif task == "Translation (Multilingual)":
    st.markdown("### 🌍 Multilingual Translation")
    lang_pairs = {
        "English to French": "Helsinki-NLP/opus-mt-en-fr",
        "English to German": "Helsinki-NLP/opus-mt-en-de",
        "English to Spanish": "Helsinki-NLP/opus-mt-en-es",
        "English to Hindi": "Helsinki-NLP/opus-mt-en-hi",
        "French to English": "Helsinki-NLP/opus-mt-fr-en",
        "German to English": "Helsinki-NLP/opus-mt-de-en",
        "Spanish to English": "Helsinki-NLP/opus-mt-es-en",
        "Hindi to English": "Helsinki-NLP/opus-mt-hi-en"
    }
    selected_pair = st.selectbox("Choose language pair", list(lang_pairs.keys()))
    if st.button("Translate"):
        if text.strip():
            translator = get_pipeline("translation", model=lang_pairs[selected_pair])
            result = translator(text)[0]['translation_text']
            st.write("**Translated Text:**")
            st.success(result)
        else:
            st.warning("Please enter text to translate.")

elif task == "Question Answering":
    context = st.text_area("Enter context (paragraph)", height=150)
    question = st.text_input("Enter your question")
    if st.button("Answer"):
        if context.strip() and question.strip():
            qa = get_pipeline("question-answering")
            answer = qa(question=question, context=context)['answer']
            st.success(f"**Answer:** {answer}")
        else:
            st.warning("Please provide both context and question.")

elif task == "Grammar Correction":
    if st.button("Correct Grammar"):
        if text.strip():
            corrector = get_pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
            result = corrector(text)[0]['generated_text']
            st.write("**Corrected Text:**")
            st.success(result)
        else:
            st.warning("Please enter text for grammar correction.")

elif task == "Text Classification":
    if st.button("Classify Text"):
        if text.strip():
            classifier = get_pipeline("text-classification")
            result = classifier(text)[0]
            st.success(f"**Label**: {result['label']} with score {round(result['score'], 2)}")
        else:
            st.warning("Please enter text for classification.")

elif task == "Language Detection":
    if st.button("Detect Language"):
        if text.strip():
            language = detect(text)
            st.success(f"**Detected Language**: {language}")
        else:
            st.warning("Please enter text to detect language.")

elif task == "Keyword Extraction":
    if st.button("Extract Keywords"):
        if text.strip():
            ner = get_pipeline("ner", grouped_entities=True)
            keywords = [ent['word'] for ent in ner(text)]
            st.write("**Keywords:**")
            st.info(list(set(keywords)))
        else:
            st.warning("Please enter text to extract keywords.")

elif task == "Chat with a Document (RAG-based)":
    st.markdown("This is a simulated RAG-based system.")
    doc = st.text_area("Paste document content")
    query = st.text_input("Ask a question about the document")
    if st.button("Get Answer"):
        if doc.strip() and query.strip():
            qa = get_pipeline("question-answering")
            answer = qa(question=query, context=doc)['answer']
            st.success(f"**Answer:** {answer}")
        else:
            st.warning("Please provide both document and question.")

# --------------------
# Example Inputs
# --------------------
with st.expander("🔍 Example Inputs"):
    st.markdown("""
    - **Sentiment:** I love using Streamlit with transformers!
    - **Summary:** The COVID-19 pandemic has affected the global economy significantly...
    - **NER:** Elon Musk founded SpaceX and Tesla.
    - **Translate:** Hello, how are you?
    - **QA:** Context: The moon is the Earth's only natural satellite. Question: What is the moon?
    - **Grammar:** she no went to school yesterdays
    - **Classify:** This movie was thrilling and suspenseful.
    - **Language Detection:** C'est une belle journée.
    - **Keywords:** Microsoft Corporation announced earnings...
    - **Chat with Doc:** Paste a news article and ask "Who is the author?"
    """)
