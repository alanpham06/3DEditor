from openai import OpenAI
from dotenv import load_dotenv
import json
import os

from glb_processor import (
    extract_scene_state,
    inspect_glb,    
    process_parameter_edits
)


# Loads the OpenAI API key from the .env file for security purposes
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_model_scene_view(scene_state):
    editable_targets = []

    for material in scene_state.get("materials", []):
        editable_targets.append({
            "id": material["id"],
            "kind": "material",
            "name": material.get("name"),
            "allowed_properties": list(material.get("editable", {}).keys()),
            "current_values": material.get("editable", {})
        })

    for node in scene_state.get("nodes", []):
        editable_targets.append({
            "id": node["id"],
            "kind": "node",
            "name": node.get("name"),
            "allowed_properties": list(node.get("editable", {}).keys()),
            "current_values": node.get("editable", {})
        })

    return {
        "editable_targets": editable_targets
    }


def build_session_context(edit_history):
    if not edit_history:
        return {
            "has_prior_edits": False,
            "entries": []
        }

    history_entries = []
    for entry in edit_history[-10:]:
        history_entries.append({
            "step": entry.get("step"),
            "user_request": entry.get("request"),
            "summary": entry.get("summary", {}),
            "applied_operations": entry.get("applied_operations", [])
        })

    return {
        "has_prior_edits": True,
        "entries": history_entries
    }

def convert_model_output_to_edit_plan(model_output):
    operations = []

    for op in model_output.get("operations", []):
        value_fields = [
            op.get("value_number", None) is not None,
            op.get("value_bool", None) is not None,
            op.get("value_array", None) is not None,
        ]

        if sum(value_fields) != 1:
            raise ValueError(
                f"Operation must set exactly one value field: {op}"
            )

        if op.get("value_number", None) is not None:
            value = op["value_number"]
        elif op.get("value_bool", None) is not None:
            value = op["value_bool"]
        else:
            value = op["value_array"]

        operations.append({
            "op": op["op"],
            "target_id": op["target_id"],
            "property": op["property"],
            "value": value
        })

    return {
        "operations": operations
    }

EDIT_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "scene_edit_plan",
    "schema": {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": [
                                "set_material_property",
                                "set_node_transform"
                            ]
                        },
                        "target_id": {"type": "string"},
                        "property": {
                            "type": "string",
                            "enum": [
                                "base_color",
                                "roughness",
                                "metallic",
                                "emissive_factor",
                                "double_sided",
                                "translation",
                                "rotation",
                                "scale"
                            ]
                        },
                        "value_number": {
                            "type": ["number", "null"]
                        },
                        "value_bool": {
                            "type": ["boolean", "null"]
                        },
                        "value_array": {
                            "type": ["array", "null"],
                            "items": {"type": "number"}
                        },
                        "rationale": {
                            "type": ["string", "null"]
                        }
                    },
                    "required": [
                        "op",
                        "target_id",
                        "property",
                        "value_number",
                        "value_bool",
                        "value_array",
                        "rationale"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": ["operations"],
        "additionalProperties": False
    },
    "strict": True
}

def request_edit_plan_from_openai(
    user_request,
    scene_state,
    edit_history=None,
    model="gpt-5.4-mini"
):
    model_scene_view = build_model_scene_view(scene_state)
    session_context = build_session_context(edit_history or [])

    system_prompt = """
You are a 3D parameter edit planner.

Your job is to convert a user request into a valid edit plan.

Rules:
- Return only operations supported by the schema.
- Use only the provided target IDs.
- Use only supported properties for each target.
- Do not invent targets.
- Do not describe textures, geometry changes, or new assets.
- Only produce parameter edits for:
  - materials: base_color, roughness, metallic, emissive_factor, double_sided
  - nodes: translation, rotation, scale
- For base_color use 4 floats.
- For emissive_factor use 3 floats.
- For translation use 3 floats.
- For rotation use 4 floats.
- For scale use 3 floats.
- Populate exactly one of:
  - value_number
  - value_bool
  - value_array
- The provided scene describes the current GLB state after any earlier edits.
- If session history is present, treat those earlier edits as already applied.
- Build on prior edits unless the user explicitly asks to undo or replace them.
- If the request cannot be expressed with supported parameter edits, return:
  { "operations": [] }
"""

    user_payload = {
        "user_request": user_request,
        "scene": model_scene_view,
        "session_history": session_context
    }

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, indent=2)
            }
        ],
        text={
            "format": EDIT_PLAN_RESPONSE_FORMAT
        }
    )

    raw_text = response.output_text
    if not raw_text:
        raise ValueError("Model returned empty structured output.")

    model_output = json.loads(raw_text)
    edit_plan = convert_model_output_to_edit_plan(model_output)

    return {
        "model_output": model_output,
        "edit_plan": edit_plan
    }


def apply_edit_plan_to_glb(
    base_glb_path,
    output_path,
    edit_plan,
):
    scene_state = extract_scene_state(base_glb_path)

    process_result = process_parameter_edits(
        original_glb_path=base_glb_path,
        scene_state=scene_state,
        edit_plan=edit_plan,
        output_path=output_path
    )

    updated_scene_state = extract_scene_state(output_path)

    return {
        "base_glb_path": base_glb_path,
        "updated_glb_path": output_path,
        "scene_state": scene_state,
        "updated_scene_state": updated_scene_state,
        "edit_plan": edit_plan,
        "process_result": process_result,
        "saved_glb": inspect_glb(output_path)
    }

def openai_parameter_edit_pipeline(
    user_request,
    original_glb_path,
    output_path,
    edit_history=None,
    model="gpt-5.4-mini"
):
    scene_state = extract_scene_state(original_glb_path)

    planning_result = request_edit_plan_from_openai(
        user_request=user_request,
        scene_state=scene_state,
        edit_history=edit_history,
        model=model
    )

    apply_result = apply_edit_plan_to_glb(
        base_glb_path=original_glb_path,
        output_path=output_path,
        edit_plan=planning_result["edit_plan"],
    )
    apply_result["user_request"] = user_request
    apply_result["model_output"] = planning_result["model_output"]
    return apply_result

def generate_updated_glb_for_viewer(
    user_request,
    original_glb_path,
    output_glb_path,
    edit_history=None,
    model="gpt-5.4-mini"
):
    # 1. Extract current editable scene state from the real GLB
    scene_state = extract_scene_state(original_glb_path)

    # 2. Ask OpenAI for a structured edit plan
    planning_result = request_edit_plan_from_openai(
        user_request=user_request,
        scene_state=scene_state,
        edit_history=edit_history,
        model=model
    )

    # 3. Apply the edit plan back into the real GLB and save it
    apply_result = apply_edit_plan_to_glb(
        base_glb_path=original_glb_path,
        output_path=output_glb_path,
        edit_plan=planning_result["edit_plan"],
    )
    apply_result["model_output"] = planning_result["model_output"]
    return apply_result

def run_openai_integration_test():
    original_glb = "objects/mask.glb"
    output_glb = "objects/updated_mask_openai.glb"

    result = openai_parameter_edit_pipeline(
        user_request="Make the helmet more metallic, slightly smoother, and scale it up by about 10 percent.",
        original_glb_path=original_glb,
        output_path=output_glb,
        model="gpt-5.4-mini"
    )

    print("MODEL OUTPUT")
    print(json.dumps(result["model_output"], indent=2))

    print("NORMALIZED EDIT PLAN")
    print(json.dumps(result["edit_plan"], indent=2))

    print("PROCESS RESULT")
    print(json.dumps(result["process_result"], indent=2))

    print("SAVED GLB INSPECTION")
    print(json.dumps(result["saved_glb"], indent=2))

if __name__ == "__main__":
    run_openai_integration_test()
