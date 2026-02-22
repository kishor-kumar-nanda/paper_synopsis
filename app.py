import streamlit as st
import base64
from pipeline.graph import build_app
from pipeline.pdf_utils import (
    pdf_to_base64_images,
    extract_page_texts,
    build_context
)

# 1. Config
st.set_page_config(page_title="Paper Synopsis", layout="wide")

# 2. Header (Fixed alignment using Markdown, as st.title doesn't support 'text_alignment')
st.markdown("<h1 style='text-align: center;'>Paper Synopsis</h1>", unsafe_allow_html=True)

uploaded_pdf = st.file_uploader("Upload Research PDF", type=["pdf"])

manual_override = st.checkbox("Manually override context (advanced)")

manual_context = ""
if manual_override:
    manual_context = st.text_area(
        "Paste custom context (figure caption / section)",
        height=200
    )

if uploaded_pdf:
    # 3. Button (Fixed: 'width' is deprecated, used 'use_container_width')
    if st.button("Run Agent", use_container_width=True):
        
        with st.spinner("Parsing PDF..."):
            pdf_bytes = uploaded_pdf.read()
            images = pdf_to_base64_images(pdf_bytes)
            page_texts = extract_page_texts(pdf_bytes)

        # Build Graph
        app = build_app()

        # Process each page
        for idx, img in enumerate(images):
            st.divider()
            st.subheader(f"Page {idx + 1}")

            context = build_context(page_texts[idx])
            context_text = manual_context or context

            # 4. Display the Image (Crucial: User needs to see what AI is looking at)
            st.image(
                f"data:image/png;base64,{img}", 
                caption=f"Page {idx+1} Input", 
                use_container_width=True
            )

            # 5. Context Preview (Fixed: Removed reference to undefined 'final' variable)
            with st.expander(f"Extracted Context (Preview)"):
                st.write(context_text[:500] + "...")

            # 6. Prepare State
            initial_state = {
                "page_text": page_texts[idx],
                "image_base64": img,
                "context_text": context_text,
                "text_understanding": None,
                "diagram_description": "",
                "critique_history": [],
                "retry_count": 0,
                "score_history": [],
                "final_quality_score": 0,
                "vision_confidence": 1.0,
                "exit_reason": "processing"
            }

            # 7. Run Agent
            with st.spinner(f"Analyzing Diagram on Page {idx + 1}..."):
                final = app.invoke(initial_state)

            # 8. Display Output
            st.markdown("### 🧠 Diagram Interpretation")
            
            # Check if 'final_explanation' exists (it might be missing if confidence was low)
            if final.get("final_explanation"):
                st.success(final["final_explanation"])
            else:
                st.warning(f"Pipeline exited early. Reason: {final.get('exit_reason', 'unknown')}")

            # 9. Debug Info
            with st.expander("Debug Details"):
                st.write(f"**Vision Description:** {final.get('diagram_description')}")
                st.write(f"**Quality Score:** {final.get('final_quality_score')}/10")
                st.write(f"**Score History:** {final.get('score_history')}")
                st.write(f"**Critiques:** {final.get('critique_history')}")