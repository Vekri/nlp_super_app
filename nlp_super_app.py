import streamlit as st
from transformers import pipeline
from langdetect import detect
import os

st.set_page_config(page_title="🧠 All-in-One NLP App", layout="wide")
st.title("🧠 Natural Language Processing Toolkit")

# Task Selector
task = st.selectbox("Choose an NLP Task", [
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
])

# Input text for all tasks (except QA and RAG)
if task not in ["Question Answering", "Chat with a Document (RAG-based)"]:
    text = st.text_area("Enter your text", height=200)

# Sentiment Analysis
if task == "Sentiment Analysis":
    if st.button("Analyze Sentiment"):
        analyzer = pipeline("sentiment-analysis")
        result = analyzer(text)[0]
        st.write(f"**Label**: {result['label']}, **Score**: {round(result['score'], 2)}")

# Text Summarization
elif task == "Text Summarization":
    if st.button("Summarize"):
        summarizer = pipeline("summarization")
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        st.write("**Summary:**")
        st.write(summary)

# Named Entity Recognition (NER)
elif task == "Named Entity Recognition (NER)":
    if st.button("Extract Entities"):
        ner = pipeline("ner", grouped_entities=True)
        entities = ner(text)
        for ent in entities:
            st.write(f"**{ent['entity_group']}**: {ent['word']} ({round(ent['score'], 2)})")

# Translation (Multilingual)
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
        translator = pipeline("translation", model=lang_pairs[selected_pair])
        result = translator(text)[0]['translation_text']
        st.write("**Translated Text:**")
        st.write(result)

# Question Answering
elif task == "Question Answering":
    context = st.text_area("Enter context (paragraph)", height=150)
    question = st.text_input("Enter your question")
    if st.button("Answer"):
        qa = pipeline("question-answering")
        answer = qa(question=question, context=context)['answer']
        st.write("**Answer:**")
        st.write(answer)

# Grammar Correction (Using text2text-generation)
elif task == "Grammar Correction":
    if st.button("Correct Grammar"):
        corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
        result = corrector(text)[0]['generated_text']
        st.write("**Corrected Text:**")
        st.write(result)

# Text Classification
elif task == "Text Classification":
    if st.button("Classify Text"):
        classifier = pipeline("text-classification")
        result = classifier(text)[0]
        st.write(f"**Label**: {result['label']} with score {round(result['score'], 2)}")

# Language Detection
elif task == "Language Detection":
    if st.button("Detect Language"):
        language = detect(text)
        st.write(f"**Detected Language**: {language}")

# Keyword Extraction (basic using NER)
elif task == "Keyword Extraction":
    if st.button("Extract Keywords"):
        ner = pipeline("ner", grouped_entities=True)
        keywords = [ent['word'] for ent in ner(text)]
        st.write("**Keywords:**")
        st.write(list(set(keywords)))

# Chat with a Document (RAG-based simulation)
elif task == "Chat with a Document (RAG-based)":
    st.markdown("This requires vector DB and advanced setup. Simulated here.")
    doc = st.text_area("Paste document content")
    query = st.text_input("Ask a question about the document")
    if st.button("Get Answer"):
        qa = pipeline("question-answering")
        answer = qa(question=query, context=doc)['answer']
        st.write("**Answer:**")
        st.write(answer)

# Examples for testing
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
