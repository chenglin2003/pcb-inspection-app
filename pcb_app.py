import streamlit as st
import analyze_pcb

st.set_page_config(page_title="PCB Defect Inspector", layout="wide")
st.title("🔍 PCB Inspection: Roboflow + AI Vision Models")

# --- Sidebar Controls ---
st.sidebar.header("Segmentation Settings")
conf_level = st.sidebar.slider("Confidence Threshold (%)", 0, 100, 40, help="Lower to see more candidates (min 9% recommended)")
overlap_level = st.sidebar.slider("Overlap (IoU) Threshold (%)", 0, 100, 45, help="Lower to reduce overlapping boxes")

def cache_uploaded_image(uploaded_file, state_prefix: str, temp_filename: str):
    signature = f"{uploaded_file.name}:{uploaded_file.size}"
    sig_key = f"{state_prefix}_signature"
    if st.session_state.get(sig_key) == signature:
        return

    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner(f"Preparing {state_prefix} image..."):
        analyze_pcb.preprocess_image(temp_filename)

    with st.spinner(f"Uploading {state_prefix} image to cloud..."):
        public_url = analyze_pcb.upload_to_gdrive(temp_filename)

    st.session_state[sig_key] = signature
    st.session_state[f"{state_prefix}_path"] = temp_filename
    st.session_state[f"{state_prefix}_url"] = public_url
    if state_prefix == "test":
        st.session_state.pop("rf_results", None)
        st.session_state.pop("rf_results_for", None)


uploaded = st.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png"], key="test_pcb_uploader")

if uploaded:
    cache_uploaded_image(uploaded, "test", "temp_pcb.png")
    if st.session_state.get("rf_results_for") != st.session_state.get("test_signature"):
        with st.spinner("Auto-detecting components on uploaded PCB image..."):
            st.session_state.rf_results = analyze_pcb.run_roboflow_inference_url(
                st.session_state.test_url, conf_level, overlap_level
            )
        st.session_state.rf_results_for = st.session_state.get("test_signature")

    # Action Buttons
    col_btn1, col_btn2 = st.columns(2)
    run_detection = col_btn1.button("1. Update Detections / Preview")
    run_ai = col_btn2.button("2. Send to GPT-4 Vision (AI Compare)")

    if run_ai:
        st.session_state.ask_for_golden = True

    if st.session_state.get("ask_for_golden"):
        st.info("Upload a Golden Reference PCB image to compare against the test image.")
        golden_uploaded = st.file_uploader(
            "Upload Golden Reference Image",
            type=["jpg", "jpeg", "png"],
            key="golden_pcb_uploader",
        )
        if golden_uploaded:
            cache_uploaded_image(golden_uploaded, "golden", "temp_golden.png")

    if "test_path" in st.session_state:
        st.subheader("Uploaded Images")
        if "golden_path" in st.session_state:
            left_col, right_col = st.columns(2)
            with left_col:
                if "rf_results" in st.session_state:
                    preview_test_path = analyze_pcb.draw_annotations(
                        st.session_state.test_path, st.session_state.rf_results["predictions"]
                    )
                    st.image(preview_test_path, caption="Test PCB (Labelled)", use_container_width=True)
                else:
                    st.image(st.session_state["test_path"], caption="Test PCB", use_container_width=True)
            with right_col:
                st.image(st.session_state["golden_path"], caption="Golden Reference PCB", use_container_width=True)
        else:
            if "rf_results" in st.session_state:
                preview_test_path = analyze_pcb.draw_annotations(
                    st.session_state.test_path, st.session_state.rf_results["predictions"]
                )
                st.image(preview_test_path, caption="Test PCB (Labelled)", use_container_width=True)
            else:
                st.image(st.session_state["test_path"], caption="Test PCB", use_container_width=True)

    # Container for Visuals
    if run_detection or 'rf_results' in st.session_state:
        if run_detection:
            with st.spinner("Scanning PCB with custom thresholds..."):
                st.session_state.rf_results = analyze_pcb.run_roboflow_inference_url(
                    st.session_state.test_url, conf_level, overlap_level
                )
            st.session_state.rf_results_for = st.session_state.get("test_signature")
        
        rf_results = st.session_state.rf_results
        
        st.subheader("JSON Labels")
        st.write(f"Found {len(rf_results['predictions'])} candidates")
        st.json(rf_results['predictions'])

    st.divider()

    # STAGE 3: Semantic Explanation
    if run_ai:
        if "golden_url" not in st.session_state or "golden_path" not in st.session_state:
            st.error("Golden reference image is required. Upload it above, then press Send to Vision Pro 4 again.")
        else:
            with st.spinner("Comparing test PCB vs golden reference and boxing differences..."):
                _, diff_boxes = analyze_pcb.compare_images_and_draw_differences(
                    st.session_state.test_path, st.session_state.golden_path
                )

            with st.spinner("Vision Pro 4 Performing Comparison Analysis..."):
                explanation = analyze_pcb.get_vision_pro_comparison_explanation(
                    st.session_state.test_url,
                    st.session_state.golden_url,
                    diff_boxes,
                    st.session_state.get("rf_results", {}).get("predictions", []),
                )
            st.subheader("🤖 Vision Pro 4 Comparison Explanation")
            st.success(explanation)
