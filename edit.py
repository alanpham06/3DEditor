import base64
import hashlib
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from glb_processor import extract_scene_state
from openai_utils import openai_parameter_edit_pipeline

APP_DIR = Path(__file__).resolve().parent
OBJECTS_DIR = APP_DIR / "objects"
SESSION_ROOT = OBJECTS_DIR / "sessions"
VIEWER_HEIGHT = 500

# Keep helper modules writing into the app folder even when Streamlit starts
# from the repo root.
os.chdir(APP_DIR)
OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
SESSION_ROOT.mkdir(parents=True, exist_ok=True)


def build_upload_key(filename: str, file_bytes: bytes) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()[:12]
    safe_name = Path(filename).stem.replace(" ", "_")
    return f"{safe_name}_{digest}"


def initialize_editor_session(uploaded_file) -> dict:
    file_bytes = uploaded_file.getvalue()
    filename = Path(uploaded_file.name).name
    upload_key = build_upload_key(filename, file_bytes)

    existing_session = st.session_state.get("editor_session")
    if existing_session and existing_session.get("upload_key") == upload_key:
        original_path = Path(existing_session["original_model_path"])
        current_path = Path(existing_session["current_model_path"])
        if original_path.exists() and current_path.exists():
            st.session_state["scene_state"] = extract_scene_state(str(current_path))
            if existing_session.get("history"):
                st.session_state["pipeline_result"] = (
                    existing_session["history"][-1].get("pipeline_result")
                )
            return existing_session

    session_dir = SESSION_ROOT / upload_key
    session_dir.mkdir(parents=True, exist_ok=True)

    original_model_path = session_dir / filename
    original_model_path.write_bytes(file_bytes)

    session = {
        "upload_key": upload_key,
        "session_dir": str(session_dir),
        "original_model_path": str(original_model_path),
        "current_model_path": str(original_model_path),
        "history": [],
        "version_counter": 0,
    }

    st.session_state["editor_session"] = session
    st.session_state["scene_state"] = extract_scene_state(str(original_model_path))
    st.session_state.pop("pipeline_result", None)
    st.session_state.pop("last_processing_seconds", None)

    return session


def next_output_glb_path(session: dict) -> tuple[Path, int]:
    next_version = session.get("version_counter", 0) + 1
    original_name = Path(session["original_model_path"]).name
    output_path = Path(session["session_dir"]) / f"step_{next_version:03d}_{original_name}"
    return output_path, next_version


def glb_to_data_url(model_path: Path) -> str:
    encoded_model = base64.b64encode(model_path.read_bytes()).decode("utf-8")
    return f"data:model/gltf-binary;base64,{encoded_model}"


def local_gltf_viewer(model_path: Path, *, height: int = VIEWER_HEIGHT) -> None:
    if not model_path.exists():
        st.warning(f"Model not found: {model_path.name}")
        return

    render_html = f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"></script>
    <model-viewer
        src="{glb_to_data_url(model_path)}"
        camera-controls
        auto-rotate
        shadow-intensity="1"
        exposure="1"
        style="width: 100%; height: {height}px; background-color: #262730; border-radius: 12px;">
    </model-viewer>
    """
    components.html(render_html, height=height + 20)


def viewer_placeholder(message: str, *, height: int = VIEWER_HEIGHT) -> None:
    placeholder_html = f"""
    <div style="
        height: {height}px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 24px;
        box-sizing: border-box;
        color: #d1d5db;
        background: #262730;
        border: 1px dashed rgba(255, 255, 255, 0.18);
        border-radius: 12px;
        font-family: sans-serif;
        font-size: 14px;
    ">
        {message}
    </div>
    """
    components.html(placeholder_html, height=height + 20)


def comparison_viewers(
    original_model_path: Path,
    updated_model_path: Path | None,
    *,
    height: int = VIEWER_HEIGHT,
) -> None:
    if not original_model_path.exists():
        st.warning(f"Model not found: {original_model_path.name}")
        return

    input_col, output_col = st.columns(2, gap="large")

    with input_col:
        st.markdown("#### Initial Input")
        st.caption(original_model_path.name)
        local_gltf_viewer(original_model_path, height=height)

    with output_col:
        st.markdown("#### Current Session Output")
        if updated_model_path and updated_model_path.exists():
            st.caption(updated_model_path.name)
            local_gltf_viewer(updated_model_path, height=height)
        else:
            st.caption("No session edits yet")
            viewer_placeholder(
                "Process the model to preview the latest session result here.",
                height=height,
            )


def summarize_process_result(process_result: dict) -> dict:
    validation_report = process_result.get("validation_report", {})
    apply_report = process_result.get("apply_report", {})
    return {
        "valid_operations": len(validation_report.get("valid_operations", [])),
        "rejected_operations": len(validation_report.get("rejected_operations", [])),
        "applied_operations": len(apply_report.get("applied_operations", [])),
        "failed_operations": len(apply_report.get("failed_operations", [])),
    }


def format_elapsed_time(elapsed_seconds: float) -> str:
    if elapsed_seconds < 1:
        return f"{elapsed_seconds * 1000:.0f} ms"
    return f"{elapsed_seconds:.2f} seconds"


def build_history_entry(
    *,
    step_number: int,
    version_number: int,
    user_prompt: str,
    result: dict,
    elapsed_seconds: float,
) -> dict:
    process_result = result.get("process_result", {})
    validation_report = process_result.get("validation_report", {})
    apply_report = process_result.get("apply_report", {})

    return {
        "step": step_number,
        "version": version_number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
        "request": user_prompt,
        "elapsed_seconds": elapsed_seconds,
        "summary": summarize_process_result(process_result),
        "applied_operations": apply_report.get("applied_operations", []),
        "failed_operations": apply_report.get("failed_operations", []),
        "rejected_operations": validation_report.get("rejected_operations", []),
        "output_model_path": result.get("updated_glb_path"),
        "pipeline_result": result,
    }


def describe_history_status(summary: dict) -> str:
    if summary.get("applied_operations", 0) > 0:
        return "Applied"
    if summary.get("valid_operations", 0) == 0:
        return "No Supported Edits"
    return "No Changes Applied"


def render_pipeline_debug(result: dict) -> None:
    scene_state = result.get("updated_scene_state") or result.get("scene_state", {})
    process_result = result.get("process_result", {})
    summary = summarize_process_result(process_result)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Valid Ops", summary["valid_operations"])
    metric_cols[1].metric("Rejected Ops", summary["rejected_operations"])
    metric_cols[2].metric("Applied Ops", summary["applied_operations"])
    metric_cols[3].metric("Failed Ops", summary["failed_operations"])

    with st.expander("Current Scene State", expanded=False):
        st.caption(
            f"Materials: {len(scene_state.get('materials', []))} | "
            f"Nodes: {len(scene_state.get('nodes', []))}"
        )
        st.json(scene_state)

    with st.expander("Model Output", expanded=True):
        st.json(result.get("model_output", {}))

    with st.expander("Normalized Edit Plan", expanded=True):
        st.json(result.get("edit_plan", {}))

    with st.expander("Process Result", expanded=True):
        st.json(process_result)

    with st.expander("Saved GLB Inspection", expanded=False):
        st.json(result.get("saved_glb", {}))


def render_session_history(session: dict) -> None:
    history = session.get("history", [])

    if not history:
        st.caption(
            "No edits yet. Each processed run is added here, and new prompts "
            "build on the latest session output."
        )
        return

    for entry in reversed(history):
        summary = entry.get("summary", {})
        label = (
            f"Step {entry['step']} | {describe_history_status(summary)} | "
            f"{entry['timestamp']}"
        )
        with st.expander(label, expanded=entry["step"] == history[-1]["step"]):
            st.markdown(f"**Request:** {entry['request']}")
            st.caption(
                f"Snapshot: {Path(entry['output_model_path']).name} | "
                f"Elapsed: {format_elapsed_time(entry['elapsed_seconds'])}"
            )

            metric_cols = st.columns(4)
            metric_cols[0].metric("Valid Ops", summary["valid_operations"])
            metric_cols[1].metric("Rejected Ops", summary["rejected_operations"])
            metric_cols[2].metric("Applied Ops", summary["applied_operations"])
            metric_cols[3].metric("Failed Ops", summary["failed_operations"])

            if entry.get("applied_operations"):
                st.markdown("**Applied Operations**")
                st.json(entry["applied_operations"])

            if entry.get("rejected_operations"):
                st.markdown("**Rejected Operations**")
                st.json(entry["rejected_operations"])

            if entry.get("failed_operations"):
                st.markdown("**Failed Operations**")
                st.json(entry["failed_operations"])


st.set_page_config(page_title="Active 3D Editor", layout="wide")
st.title("Active 3D Editor")
st.caption(
    "Upload a GLB, describe parameter edits, and compare the original model "
    "with the latest cumulative session result."
)

uploaded_file = st.file_uploader("Upload your .glb model", type=["glb"])

if uploaded_file:
    session = initialize_editor_session(uploaded_file)
    original_model_path = Path(session["original_model_path"])
    current_model_path = Path(session["current_model_path"])

    st.success(f"Loaded {original_model_path.name}")
    st.caption(
        f"Session edits: {len(session['history'])}. New requests build on "
        f"`{current_model_path.name}`."
    )

    user_prompt = st.text_area(
        "What modifications should the AI make next?",
        placeholder=(
            "e.g. Make the first material more metallic and slightly smoother, "
            "then scale the main node up by 10%."
        ),
    )

    with st.expander("Current Editable Scene State", expanded=False):
        st.json(st.session_state["scene_state"])

    process_col, undo_col = st.columns([3, 1])
    with process_col:
        process_clicked = st.button(
            "Process Model",
            type="primary",
            use_container_width=True,
        )
    with undo_col:
        undo_clicked = st.button(
            "Undo Last Change",
            disabled=not session["history"],
            use_container_width=True,
        )

    if undo_clicked:
        removed_entry = session["history"].pop()

        if session["history"]:
            restored_entry = session["history"][-1]
            session["current_model_path"] = restored_entry["output_model_path"]
            st.session_state["pipeline_result"] = restored_entry["pipeline_result"]
        else:
            session["current_model_path"] = session["original_model_path"]
            st.session_state.pop("pipeline_result", None)

        current_model_path = Path(session["current_model_path"])
        st.session_state["editor_session"] = session
        st.session_state["scene_state"] = extract_scene_state(str(current_model_path))
        st.session_state.pop("last_processing_seconds", None)

        st.success(
            f"Removed step {removed_entry['step']}. The current session model is "
            f"now `{current_model_path.name}`."
        )

    if process_clicked:
        if not user_prompt.strip():
            st.error("Add a modification request so the AI knows what to change.")
        else:
            output_path, version_number = next_output_glb_path(session)
            start_time = time.perf_counter()

            try:
                with st.spinner(
                    "AI is planning parameter edits from the latest session "
                    "state and applying them..."
                ):
                    result = openai_parameter_edit_pipeline(
                        user_request=user_prompt.strip(),
                        original_glb_path=str(current_model_path),
                        output_path=str(output_path),
                        edit_history=session["history"],
                    )
            except Exception as exc:
                st.session_state["last_processing_seconds"] = (
                    time.perf_counter() - start_time
                )
                st.error(f"Processing failed: {exc}")
            else:
                elapsed_seconds = time.perf_counter() - start_time
                history_entry = build_history_entry(
                    step_number=len(session["history"]) + 1,
                    version_number=version_number,
                    user_prompt=user_prompt.strip(),
                    result=result,
                    elapsed_seconds=elapsed_seconds,
                )

                session["history"].append(history_entry)
                session["current_model_path"] = str(output_path)
                session["version_counter"] = version_number
                current_model_path = output_path

                st.session_state["editor_session"] = session
                st.session_state["scene_state"] = result.get("updated_scene_state", {})
                st.session_state["pipeline_result"] = result
                st.session_state["last_processing_seconds"] = elapsed_seconds

                summary = history_entry["summary"]
                if summary["applied_operations"] > 0:
                    st.balloons()
                    st.success(
                        "Done! The session advanced to "
                        f"`{output_path.resolve()}`"
                    )
                elif summary["valid_operations"] == 0:
                    st.warning(
                        "The AI did not produce any supported parameter edits. "
                        "A new session snapshot was still written for inspection."
                    )
                else:
                    st.warning(
                        "The edit plan was generated, but no operations were "
                        "applied successfully. Check the reports below."
                    )

    elapsed_seconds = st.session_state.get("last_processing_seconds")
    if elapsed_seconds is not None:
        st.info(
            "Processing time from clicking `Process Model` to AI completion: "
            f"{format_elapsed_time(elapsed_seconds)}"
        )

    st.subheader("3D Model Viewer")
    comparison_viewers(
        original_model_path,
        current_model_path if session["history"] else None,
    )

    st.subheader("Session History")
    render_session_history(session)

    pipeline_result = st.session_state.get("pipeline_result")
    if pipeline_result:
        st.subheader("Latest Edit Planning Details")
        render_pipeline_debug(pipeline_result)
