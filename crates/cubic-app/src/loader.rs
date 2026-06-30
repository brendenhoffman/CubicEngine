// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::{Context, Result};
use cubic_render_vk::Vertex;
use std::path::Path;

/// Load an OBJ file and return (vertices, indices) ready to pass directly to
/// `VkRenderer::upload_mesh`. All sub-meshes in the file are merged into one
/// flat list so the whole file is a single draw call.
///
/// OBJ has no per-vertex colour; `Vertex::color` defaults to white so the
/// `tint` field in the per-draw `PushData` can control appearance without
/// needing a separate texture. UVs and normals default to zero/up if absent
/// (e.g. an OBJ that was exported without them).
pub fn load_obj_mesh(path: &Path) -> Result<(Vec<Vertex>, Vec<u32>)> {
    let (models, _) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)
        .with_context(|| format!("load_obj {:?}", path))?;

    let mut verts: Vec<Vertex> = Vec::new();
    let mut idxs: Vec<u32> = Vec::new();

    for model in &models {
        let mesh = &model.mesh;
        let base = verts.len() as u32;
        let n = mesh.positions.len() / 3;

        for i in 0..n {
            let pos = [
                mesh.positions[i * 3],
                mesh.positions[i * 3 + 1],
                mesh.positions[i * 3 + 2],
            ];
            let normal = if mesh.normals.len() >= (i + 1) * 3 {
                [
                    mesh.normals[i * 3],
                    mesh.normals[i * 3 + 1],
                    mesh.normals[i * 3 + 2],
                ]
            } else {
                [0.0, 0.0, 1.0]
            };
            let uv = if mesh.texcoords.len() >= (i + 1) * 2 {
                [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1]]
            } else {
                [0.0, 0.0]
            };
            verts.push(Vertex {
                pos,
                color: [1.0, 1.0, 1.0], // OBJ carries no per-vertex colour
                uv,
                normal,
            });
        }

        for &idx in &mesh.indices {
            idxs.push(base + idx);
        }
    }

    Ok((verts, idxs))
}
