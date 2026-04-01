from pygltflib import GLTF2
from pathlib import Path
import os
import json

###############################################################################
#####                    glb property processor                           #####
###############################################################################

def extract_properties(file_path):
    # Load the GLB file
    gltf = GLTF2().load(file_path)
    name = Path(file_path).stem

    # Helper to resolve texture index -> image metadata
    def resolve_image_info(texture_info):
        if texture_info is None:
            return None
        tex_idx = texture_info.index
        # Get the source image index from the texture
        source_idx = gltf.textures[tex_idx].source
        img = gltf.images[source_idx]
        return {
            "image_name": img.name,
            "mime_type": img.mimeType
        }

    # Define custom format structure
    custom_data = {
        "model_metadata": gltf.asset.to_dict(),
        "objects": [],
        "materials": []
    }

    # 1. Extract Material Properties & All Texture Maps
    for material in gltf.materials:
        pbr = material.pbrMetallicRoughness
        
        mat_info = {
            "name": material.name,
            "properties": {
                "base_color": pbr.baseColorFactor,
                "roughness": pbr.roughnessFactor,
                "metallic": pbr.metallicFactor,
                "emissive_factor": material.emissiveFactor,
                "double_sided": material.doubleSided,
            },
            "texture_maps": {
                "albedo": resolve_image_info(pbr.baseColorTexture),
                "normal": resolve_image_info(material.normalTexture),
                "emissive": resolve_image_info(material.emissiveTexture),
                # Occlusion and Metal/Rough often share the same image file
                "occlusion_metallic_roughness": resolve_image_info(material.occlusionTexture or 
                                                                   pbr.metallicRoughnessTexture)
            }
        }
        custom_data["materials"].append(mat_info)

    # 2. Extract Object Nodes (Transformations and Physical Dimensions)
    for node in gltf.nodes:
        if node.mesh is not None:
            mesh = gltf.meshes[node.mesh]
            # Get dimensions from the POSITION accessor of the first primitive
            pos_accessor_idx = mesh.primitives[0].attributes.POSITION
            accessor = gltf.accessors[pos_accessor_idx]

            obj_info = {
                "node_name": node.name,
                "transformation": {
                    "translation": node.translation,
                    "rotation": node.rotation,
                    "scale": node.scale
                },
                "dimensions": {
                    "min": accessor.min,
                    "max": accessor.max
                }
            }
            custom_data["objects"].append(obj_info)

    # Save to the custom format
    output_path = f"json/{name}_properties.json" 
    with open(output_path, "w") as f:
        json.dump(custom_data, f, indent=4)
    
    print(f"Exported: {output_path}")

    return output_path

# Example Usage
# extract_properties("objects/cube_color.glb")


###############################################################################
#####                    glb texture processor                            #####
###############################################################################

def extract_textures(glb_path, output_dir="extracted_textures"):
    gltf = GLTF2().load(glb_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the raw binary chunk (the 'BIN' part of the GLB)
    binary_blob = gltf.binary_blob()

    # Create a list for all of the texture paths
    texture_paths = []

    for i, image in enumerate(gltf.images):
        if image.bufferView is not None:
            # Look up the specific slice of the binary data
            bv = gltf.bufferViews[image.bufferView]
            
            # Slice the binary blob: from byteOffset to byteOffset + byteLength
            start = bv.byteOffset
            end = start + bv.byteLength
            image_data = binary_blob[start:end]
            
            # Determine extension
            ext = "png" if "png" in (image.mimeType or "") else "jpg"
            name = image.name if image.name else f"texture_{i}"
            
            save_path = os.path.join(output_dir, f"{name}.{ext}")
            
            with open(save_path, "wb") as f:
                f.write(image_data)
            
            print(f"Saved: {save_path}")

            texture_paths.append(save_path)
    
    return texture_paths

# Run it
# extract_textures("objects/mask.glb")
