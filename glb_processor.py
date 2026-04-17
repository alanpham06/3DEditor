from pygltflib import GLTF2
from pygltflib import Material, PbrMetallicRoughness
import json
import os

MATERIAL_PROPERTIES = {
    "base_color",
    "roughness",
    "metallic",
    "emissive_factor",
    "double_sided",
}

NODE_PROPERTIES = {
    "translation",
    "rotation",
    "scale",
}

###############################################################################
#####                glb processor helper functions                       #####
###############################################################################

def build_scene_index(scene_state):
    index = {}

    for material in scene_state.get("materials", []):
        index[material["id"]] = {
            "kind": "material",
            "name": material.get("name"),
            "gltf_index": material["gltf_material_index"],
            "editable": material.get("editable", {})
        }

    for node in scene_state.get("nodes", []):
        index[node["id"]] = {
            "kind": "node",
            "name": node.get("name"),
            "gltf_index": node["gltf_node_index"],
            "editable": node.get("editable", {})
        }

    return index


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def ensure_float(value):
    return float(value)


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Cannot convert {value!r} to bool")


def normalize_float_list(value, expected_len):
    if not isinstance(value, list):
        raise ValueError(f"Expected list of length {expected_len}")
    if len(value) != expected_len:
        raise ValueError(f"Expected list of length {expected_len}, got {len(value)}")
    return [float(v) for v in value]


def normalize_material_value(prop, value):
    if prop == "roughness":
        return clamp(ensure_float(value), 0.0, 1.0)

    if prop == "metallic":
        return clamp(ensure_float(value), 0.0, 1.0)

    if prop == "base_color":
        vals = normalize_float_list(value, 4)
        return [clamp(v, 0.0, 1.0) for v in vals]

    if prop == "emissive_factor":
        vals = normalize_float_list(value, 3)
        return [max(0.0, v) for v in vals]

    if prop == "double_sided":
        return normalize_bool(value)

    raise ValueError(f"Unsupported material property: {prop}")


def normalize_node_value(prop, value):
    if prop == "translation":
        return normalize_float_list(value, 3)

    if prop == "rotation":
        return normalize_float_list(value, 4)

    if prop == "scale":
        vals = normalize_float_list(value, 3)
        if any(v <= 0 for v in vals):
            raise ValueError("Scale values must be > 0")
        return vals

    raise ValueError(f"Unsupported node property: {prop}")

def validate_and_normalize_edit_plan(edit_plan, scene_index):
    if not isinstance(edit_plan, dict):
        raise ValueError("Edit plan must be a dictionary")

    operations = edit_plan.get("operations")
    if not isinstance(operations, list):
        raise ValueError("Edit plan must contain an 'operations' list")

    normalized_ops = []
    rejected_ops = []

    for i, op in enumerate(operations):
        try:
            if not isinstance(op, dict):
                raise ValueError("Operation must be an object")

            op_type = op.get("op")
            target_id = op.get("target_id")
            prop = op.get("property")
            value = op.get("value")

            if target_id not in scene_index:
                raise ValueError(f"Unknown target_id: {target_id}")

            target = scene_index[target_id]
            kind = target["kind"]

            if op_type == "set_material_property":
                if kind != "material":
                    raise ValueError(f"Target {target_id} is not a material")
                if prop not in MATERIAL_PROPERTIES:
                    raise ValueError(f"Unsupported material property: {prop}")

                normalized_value = normalize_material_value(prop, value)

            elif op_type == "set_node_transform":
                if kind != "node":
                    raise ValueError(f"Target {target_id} is not a node")
                if prop not in NODE_PROPERTIES:
                    raise ValueError(f"Unsupported node property: {prop}")

                normalized_value = normalize_node_value(prop, value)

            else:
                raise ValueError(f"Unsupported operation type: {op_type}")

            normalized_ops.append({
                "op": op_type,
                "target_id": target_id,
                "property": prop,
                "value": normalized_value
            })

        except Exception as e:
            rejected_ops.append({
                "index": i,
                "operation": op,
                "reason": str(e)
            })

    return {
        "valid_operations": normalized_ops,
        "rejected_operations": rejected_ops
    }

def ensure_pbr(material: Material):
    if material.pbrMetallicRoughness is None:
        material.pbrMetallicRoughness = PbrMetallicRoughness()


def apply_parameter_edits(gltf, scene_index, validated_plan):
    applied = []
    failed = []

    for op in validated_plan.get("valid_operations", []):
        try:
            target = scene_index[op["target_id"]]
            gltf_index = target["gltf_index"]
            prop = op["property"]
            value = op["value"]

            if op["op"] == "set_material_property":
                material = gltf.materials[gltf_index]
                ensure_pbr(material)

                if prop == "base_color":
                    material.pbrMetallicRoughness.baseColorFactor = value

                elif prop == "roughness":
                    material.pbrMetallicRoughness.roughnessFactor = value

                elif prop == "metallic":
                    material.pbrMetallicRoughness.metallicFactor = value

                elif prop == "emissive_factor":
                    material.emissiveFactor = value

                elif prop == "double_sided":
                    material.doubleSided = value

                else:
                    raise ValueError(f"Unhandled material property: {prop}")

            elif op["op"] == "set_node_transform":
                node = gltf.nodes[gltf_index]

                if prop == "translation":
                    node.translation = value

                elif prop == "rotation":
                    node.rotation = value

                elif prop == "scale":
                    node.scale = value

                else:
                    raise ValueError(f"Unhandled node property: {prop}")

            else:
                raise ValueError(f"Unhandled operation type: {op['op']}")

            applied.append(op)

        except Exception as e:
            failed.append({
                "operation": op,
                "reason": str(e)
            })

    return {
        "applied_operations": applied,
        "failed_operations": failed
    }


def process_parameter_edits(
    original_glb_path,
    scene_state,
    edit_plan,
    output_path
):
    gltf = GLTF2().load(original_glb_path)

    scene_index = build_scene_index(scene_state)

    validation_report = validate_and_normalize_edit_plan(edit_plan, scene_index)
    apply_report = apply_parameter_edits(gltf, scene_index, validation_report)

    gltf.save(output_path)

    return {
        "output_path": output_path,
        "validation_report": validation_report,
        "apply_report": apply_report
    }


#################################################################################

def extract_scene_state(glb_path):
    gltf = GLTF2().load(glb_path)

    scene_state = {
        "materials": [],
        "nodes": []
    }

    # Materials
    for i, material in enumerate(gltf.materials or []):
        pbr = material.pbrMetallicRoughness or PbrMetallicRoughness()

        base_color = pbr.baseColorFactor if pbr.baseColorFactor is not None else [1.0, 1.0, 1.0, 1.0]
        roughness = pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0
        metallic = pbr.metallicFactor if pbr.metallicFactor is not None else 1.0
        emissive = material.emissiveFactor if material.emissiveFactor is not None else [0.0, 0.0, 0.0]
        double_sided = material.doubleSided if material.doubleSided is not None else False

        scene_state["materials"].append({
            "id": f"mat:{i}",
            "kind": "material",
            "name": material.name or f"material_{i}",
            "gltf_material_index": i,
            "editable": {
                "base_color": list(base_color),
                "roughness": float(roughness),
                "metallic": float(metallic),
                "emissive_factor": list(emissive),
                "double_sided": bool(double_sided)
            }
        })

    # Nodes
    for i, node in enumerate(gltf.nodes or []):
        translation = node.translation if node.translation is not None else [0.0, 0.0, 0.0]
        rotation = node.rotation if node.rotation is not None else [0.0, 0.0, 0.0, 1.0]
        scale = node.scale if node.scale is not None else [1.0, 1.0, 1.0]

        scene_state["nodes"].append({
            "id": f"node:{i}",
            "kind": "node",
            "name": node.name or f"node_{i}",
            "gltf_node_index": i,
            "editable": {
                "translation": [float(v) for v in translation],
                "rotation": [float(v) for v in rotation],
                "scale": [float(v) for v in scale]
            }
        })

    return scene_state

def inspect_glb(glb_path):
    gltf = GLTF2().load(glb_path)

    materials = []
    for i, material in enumerate(gltf.materials or []):
        pbr = material.pbrMetallicRoughness or PbrMetallicRoughness()
        materials.append({
            "id": f"mat:{i}",
            "name": material.name or f"material_{i}",
            "base_color": pbr.baseColorFactor if pbr.baseColorFactor is not None else [1.0, 1.0, 1.0, 1.0],
            "roughness": pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0,
            "metallic": pbr.metallicFactor if pbr.metallicFactor is not None else 1.0,
            "emissive_factor": material.emissiveFactor if material.emissiveFactor is not None else [0.0, 0.0, 0.0],
            "double_sided": material.doubleSided if material.doubleSided is not None else False
        })

    nodes = []
    for i, node in enumerate(gltf.nodes or []):
        nodes.append({
            "id": f"node:{i}",
            "name": node.name or f"node_{i}",
            "translation": node.translation if node.translation is not None else [0.0, 0.0, 0.0],
            "rotation": node.rotation if node.rotation is not None else [0.0, 0.0, 0.0, 1.0],
            "scale": node.scale if node.scale is not None else [1.0, 1.0, 1.0]
        })

    return {"materials": materials, "nodes": nodes}

###############################################################################
########                Test for the GLB Processor                        #####
###############################################################################


def run_smoke_test():
    original_glb = "objects/mask.glb"
    output_glb = "objects/updated_mask.glb"

    scene_state = extract_scene_state(original_glb)

    if not scene_state["materials"]:
        raise ValueError("No materials found in GLB")
    if not scene_state["nodes"]:
        raise ValueError("No nodes found in GLB")

    edit_plan = {
        "operations": [
            {
                "op": "set_material_property",
                "target_id": "mat:0",
                "property": "roughness",
                "value": 0.12
            },
            {
                "op": "set_material_property",
                "target_id": "mat:0",
                "property": "metallic",
                "value": 0.88
            },
            {
                "op": "set_node_transform",
                "target_id": "node:0",
                "property": "scale",
                "value": [1.15, 1.15, 1.15]
            }
        ]
    }

    report = process_parameter_edits(
        original_glb_path=original_glb,
        scene_state=scene_state,
        edit_plan=edit_plan,
        output_path=output_glb
    )

    print("PROCESS REPORT")
    print(json.dumps(report, indent=2))

    saved = inspect_glb(output_glb)

    assert abs(saved["materials"][0]["roughness"] - 0.12) < 1e-6, "roughness not updated"
    assert abs(saved["materials"][0]["metallic"] - 0.88) < 1e-6, "metallic not updated"
    assert saved["nodes"][0]["scale"] == [1.15, 1.15, 1.15], "scale not updated"

    print("Smoke test passed.")
    print(json.dumps(saved, indent=2))

def run_invalid_test():
    original_glb = "objects/mask.glb"
    scene_state = extract_scene_state(original_glb)
    scene_index = build_scene_index(scene_state)

    bad_plan = {
        "operations": [
            {
                "op": "set_material_property",
                "target_id": "mat:999",
                "property": "roughness",
                "value": 0.5
            },
            {
                "op": "set_node_transform",
                "target_id": "node:0",
                "property": "scale",
                "value": [1.0, -2.0, 1.0]
            }
        ]
    }

    result = validate_and_normalize_edit_plan(bad_plan, scene_index)
    print("INVALID TEST RESULT")
    print(json.dumps(result, indent=2))

###############################################################################
###############################################################################

if __name__ == "__main__":
    run_smoke_test()
    run_invalid_test()