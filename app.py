from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import SETTINGS
from llm.gemini_client import GeminiClient
from llm.response_generator import ResponseGenerator
from memory.extractor import extract_memories
from memory.long_term import LongTermMemoryStore
from memory.short_term import ShortTermMemory
from rag.ingest import ingest_corpus
from rag.retriever import ChromaRetriever


load_dotenv()

st.set_page_config(page_title="Digital Twin of Neil deGrasse Tyson", layout="wide")

st.title("Digital Twin of Neil deGrasse Tyson")
st.caption("RAG + memory + persona-driven science dialogue")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"
if "memory" not in st.session_state:
    st.session_state.memory = ShortTermMemory(max_messages=SETTINGS.recent_turns)

with st.sidebar:
    st.header("Controls")
    api_key = st.text_input("Gemini API key", value=st.session_state.api_key, type="password")
    st.session_state.api_key = api_key.strip()

    user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id.strip() or "default_user"

    if st.button("Build / refresh index"):
        if not st.session_state.api_key:
            st.error("Enter your Gemini API key first.")
        else:
            with st.spinner("Indexing documents from data/raw ..."):
                count = ingest_corpus(api_key=st.session_state.api_key, raw_dir=SETTINGS.raw_data_dir)
            st.success(f"Indexed {count} chunks.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.divider()
    st.write("Add text files to `data/raw/` and refresh the index.")
    st.write("The app stores durable memories in a separate Chroma collection.")

if not st.session_state.api_key:
    st.warning("Enter your Gemini API key in the sidebar to begin.")
    st.stop()

retriever = ChromaRetriever(api_key=st.session_state.api_key)
memory_store = LongTermMemoryStore(api_key=st.session_state.api_key)
generator = ResponseGenerator(
    api_key=st.session_state.api_key,
    model=SETTINGS.gemini_model,
    retriever=retriever,
    memory_store=memory_store,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about astrophysics, physics, or science in general")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.memory.add("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        assistant_text, bundle = generator.generate(
            user_id=st.session_state.user_id,
            query=user_input,
            short_term=st.session_state.memory,
        )

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.session_state.memory.add("assistant", assistant_text)

    with st.chat_message("assistant"):
        st.markdown(assistant_text)

        with st.expander("Retrieved sources"):
            if bundle.sources:
                for src in bundle.sources:
                    st.write(f"- {src['title']} | chunk {src['chunk_id']} | {src['source']} | distance={src['distance']:.4f}")
            else:
                st.write("No sources retrieved.")

        with st.expander("Relevant memories"):
            if bundle.memories:
                for mem in bundle.memories:
                    st.write(f"- ({mem['memory_type']}, importance {mem['importance']}) {mem['text']}")
            else:
                st.write("No memories retrieved.")

    if st.button("Store durable memories from this turn"):
        extractor_client = GeminiClient(api_key=st.session_state.api_key, model=SETTINGS.gemini_model)
        extracted = extract_memories(extractor_client, user_input, assistant_text)
        memory_store.add_memories(st.session_state.user_id, extracted)
        st.success(f"Stored {len(extracted)} durable memories.")
