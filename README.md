# Digital Twin of Neil deGrasse Tyson

A Streamlit-based Digital Twin project with:
- Gemini 2.5 Flash chat generation
- RAG over local text sources
- short-term memory for the current session
- long-term memory persisted across sessions
- a simple file structure that is easy to extend

## What to put in `data/raw/`
Add public text documents such as:
- `.txt`
- `.md`
- `.pdf`

Good source types for this project:
- interviews
- lecture transcripts
- public articles
- book excerpts you are allowed to use
- transcripts from talks and podcasts

## Setup

1. Create a virtual environment
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add your key to a `.env` file

```bash
cp .env.example .env
```

4. Put documents into `data/raw/`

5. Build the index

```bash
python -m rag.ingest
```

6. Run the app

```bash
streamlit run app.py
```

## Notes

- The app lets you enter the API key in the sidebar too.
- The RAG layer uses ChromaDB for persistence.
- The embedding layer uses the Google GenAI SDK.
- The memory layer stores durable user facts in a separate Chroma collection.
