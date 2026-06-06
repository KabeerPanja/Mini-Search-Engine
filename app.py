import streamlit as st
import uuid

from utils.pdf_loader import load_pdf_text
from utils.chunk import split_text
from utils.embeddings import get_embedding
from utils.pinecone_store import upsert_vectors, search_vector

st.title("📄 Mini Search Engine (Vector + Pinecone)")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("Process & Store"):
        all_vectors = []

        for file in uploaded_files:
            text = load_pdf_text(file)
            chunks = split_text(text)

            for chunk in chunks:
                emb = get_embedding(chunk)

                all_vectors.append(
                    (
                        str(uuid.uuid4()),
                        emb,
                        {
                            "text": chunk,
                            "file_name": file.name
                        }
                    )
                )

        upsert_vectors(all_vectors)
        st.success("Documents processed and stored in Pinecone!")

st.divider()

query = st.text_input("Search anything from PDFs")

if st.button("Search"):
    if query.strip():

        query_emb = get_embedding(query)
        results = search_vector(query_emb)

        st.subheader("Results")

        matches = results.get("matches", [])

        if not matches:
            st.warning("No results found.")
        else:
            for match in matches:

                metadata = match.get("metadata", {})

                file_name = metadata.get("file_name", "Unknown file")
                text = metadata.get("text", "No text available")
                score = round(match.get("score", 0), 3)

                st.write("📄 File:", file_name)
                st.write("⭐ Score:", score)
                st.write("📌 Text:", text)
                st.write("---")