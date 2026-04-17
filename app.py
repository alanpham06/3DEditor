import os
from pathlib import Path

import gradio as gr

from glb_processor import extract_properties, extract_textures
from openai_utils import openai_3DEditor

APP_DIR = Path(__file__).resolve().parent
OBJECTS_DIR = APP_DIR / "objects"

# Keep helper modules writing into the app folder even when Gradio starts
# from the repo root.
os.chdir(APP_DIR)
OBJECTS_DIR.mkdir(parents=True, exist_ok=True)

VIEWER_HEIGHT = 500
EMPTY_UPLOAD_MESSAGE = "Upload a `.glb` model to get started."
EMPTY_INPUT_MESSAGE = "Upload a GLB file to preview it here."
EMPTY_OUTPUT_MESSAGE = "Process the model to preview the AI-generated result here."


def save_uploaded_glb(uploaded_file: str) -> Path:
    source_path = Path(uploaded_file)
    model_path = OBJECTS_DIR / source_path.name
    model_path.write_bytes(source_path.read_bytes())
    return model_path


def reset_view():
    return (
        EMPTY_UPLOAD_MESSAGE,
        EMPTY_INPUT_MESSAGE,
        None,
        EMPTY_OUTPUT_MESSAGE,
        None,
        None,
        None,
        None,
    )


def load_model(uploaded_file: str | None):
    if not uploaded_file:
        return reset_view()

    original_model_path = save_uploaded_glb(uploaded_file)
    updated_model_path = OBJECTS_DIR / f"updated_{original_model_path.name}"
    updated_value = str(updated_model_path) if updated_model_path.exists() else None

    updated_message = (
        f"`{updated_model_path.name}`"
        if updated_value
        else EMPTY_OUTPUT_MESSAGE
    )

    return (
        f"Loaded `{original_model_path.name}`",
        f"`{original_model_path.name}`",
        str(original_model_path),
        updated_message,
        updated_value,
        str(original_model_path),
        updated_value,
        updated_value,
    )


def process_model(user_prompt: str, original_model_path: str | None):
    if not original_model_path:
        return (
            "Upload a model before running the editor.",
            EMPTY_OUTPUT_MESSAGE,
            None,
            None,
            None,
        )

    original_path = Path(original_model_path)
    if not original_path.exists():
        return (
            "The uploaded model could not be found. Please upload it again.",
            EMPTY_OUTPUT_MESSAGE,
            None,
            None,
            None,
        )

    if not user_prompt or not user_prompt.strip():
        return (
            "Add a modification request so the AI knows what to change.",
            EMPTY_OUTPUT_MESSAGE,
            None,
            None,
            None,
        )

    json_path = extract_properties(str(original_path))
    textures = extract_textures(str(original_path))
    result = openai_3DEditor(
        user_prompt.strip(),
        str(original_path),
        json_path,
        textures,
        output_folder=str(OBJECTS_DIR),
    )

    if not result:
        return (
            "The model update did not complete successfully.",
            EMPTY_OUTPUT_MESSAGE,
            None,
            None,
            None,
        )

    updated_path = Path(result)
    status = (
        "Done. The updated model was saved to "
        f"`{updated_path.resolve()}`"
    )
    return (
        status,
        f"`{updated_path.name}`",
        str(updated_path),
        str(updated_path),
        str(updated_path),
    )


with gr.Blocks(
    title="Active 3D Editor",
    theme=gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css="""
    .app-shell {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px 0 32px;
    }

    .hero {
        background:
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.18), transparent 35%),
            linear-gradient(135deg, #eff6ff 0%, #f8fafc 55%, #e2e8f0 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 20px;
    }

    .hero h1 {
        margin: 0;
        color: #0f172a;
        font-size: 2.2rem;
        line-height: 1.1;
    }

    .hero p {
        margin: 10px 0 0;
        color: #334155;
        font-size: 1rem;
        max-width: 720px;
    }

    .status-box {
        min-height: 74px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        background: #f8fafc;
        padding: 14px 18px;
    }

    .viewer-card {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
        min-height: 620px;
    }

    .viewer-heading h3 {
        margin: 0 0 6px 0;
        color: #f8fafc;
        font-size: 1.05rem;
    }

    .viewer-note {
        min-height: 42px;
        margin-bottom: 8px;
    }

    .viewer-note p {
        margin: 0;
        color: #94a3b8;
        font-size: 0.92rem;
        word-break: break-word;
    }

    .model-stage {
        border-radius: 14px;
        overflow: hidden;
    }
    """,
) as demo:
    original_state = gr.State(value=None)
    updated_state = gr.State(value=None)

    with gr.Column(elem_classes="app-shell"):
        gr.HTML(
            """
            <section class="hero">
                <h1>Active 3D Editor</h1>
                <p>Upload a GLB asset, describe the material edits you want, and compare the original model with the AI-generated output side by side.</p>
            </section>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                model_upload = gr.File(
                    label="Upload your .glb model",
                    file_types=[".glb"],
                    type="filepath",
                )
                user_prompt = gr.Textbox(
                    label="What modifications should the AI make?",
                    placeholder="e.g. Change the texture to rusted metal...",
                    lines=4,
                )
                process_button = gr.Button("Process Model", variant="primary")
            with gr.Column(scale=4):
                status_box = gr.Markdown(
                    "Upload a `.glb` model to get started.",
                    elem_classes="status-box",
                )
                updated_file = gr.File(
                    label="Latest AI Output",
                    interactive=False,
                )

        with gr.Row(equal_height=False):
            with gr.Column(elem_classes="viewer-card"):
                gr.Markdown("### Initial Input", elem_classes="viewer-heading")
                original_note = gr.Markdown(
                    EMPTY_INPUT_MESSAGE,
                    elem_classes="viewer-note",
                )
                original_viewer = gr.Model3D(
                    label="Initial Input Viewer",
                    show_label=False,
                    interactive=False,
                    height=VIEWER_HEIGHT,
                    clear_color=(0.07, 0.09, 0.12, 1.0),
                    elem_classes="model-stage",
                )
            with gr.Column(elem_classes="viewer-card"):
                gr.Markdown("### AI Output", elem_classes="viewer-heading")
                updated_note = gr.Markdown(
                    EMPTY_OUTPUT_MESSAGE,
                    elem_classes="viewer-note",
                )
                updated_viewer = gr.Model3D(
                    label="AI Output Viewer",
                    show_label=False,
                    interactive=False,
                    height=VIEWER_HEIGHT,
                    clear_color=(0.07, 0.09, 0.12, 1.0),
                    elem_classes="model-stage",
                )

    model_upload.change(
        fn=load_model,
        inputs=[model_upload],
        outputs=[
            status_box,
            original_note,
            original_viewer,
            updated_note,
            updated_viewer,
            original_state,
            updated_state,
            updated_file,
        ],
    )

    process_button.click(
        fn=process_model,
        inputs=[user_prompt, original_state],
        outputs=[
            status_box,
            updated_note,
            updated_viewer,
            updated_state,
            updated_file,
        ],
    )


if __name__ == "__main__":
    demo.launch()
