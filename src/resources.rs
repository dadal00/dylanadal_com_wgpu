// Standard Library
use std::{
    f32::consts::PI,
    fs::{read, read_to_string},
    io::{BufReader, Cursor},
    path::Path,
};

// External
use anyhow::Result;
use bytemuck::cast_slice;
use cgmath::{Vector2, Vector3};
use tobj::{LoadOptions, load_mtl_buf, load_obj_buf_async};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};

// Web Assembly
#[cfg(target_arch = "wasm32")]
use reqwest::{Url, get};

// Internal Modules
use crate::{
    model::{Material, Mesh, Model, ModelVertex},
    texture::Texture,
};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let mut origin = location.origin().unwrap();
    if !origin.ends_with("learn-wgpu") {
        origin = format!("{}/learn-wgpu", origin);
    }
    let base = Url::parse(&format!("{}/", origin,)).unwrap();
    base.join(file_name).unwrap()
}

pub async fn load_string(file_name: &str) -> Result<String> {
    #[cfg(target_arch = "wasm32")]
    let txt = {
        let url = format_url(file_name);
        get(url).await?.text().await?
    };

    #[cfg(not(target_arch = "wasm32"))]
    let txt = {
        let path = Path::new(env!("OUT_DIR")).join("res").join(file_name);
        read_to_string(path)?
    };

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> Result<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    let data = {
        let url = format_url(file_name);
        get(url).await?.bytes().await?.to_vec()
    };
    #[cfg(not(target_arch = "wasm32"))]
    let data = {
        let path = Path::new(env!("OUT_DIR")).join("res").join(file_name);
        read(path)?
    };

    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    is_normal_map: bool,
    device: &Device,
    queue: &Queue,
) -> Result<Texture> {
    let data = load_binary(file_name).await?;
    Texture::from_bytes(device, queue, &data, file_name, is_normal_map)
}

pub async fn load_model(
    file_name: &str,
    device: &Device,
    queue: &Queue,
    layout: &BindGroupLayout,
) -> Result<Model> {
    let obj_text = load_string(file_name).await?;

    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = load_obj_buf_async(
        &mut obj_reader,
        &LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = load_texture(&m.diffuse_texture, false, device, queue).await?;
        let normal_texture = load_texture(&m.normal_texture, true, device, queue).await?;

        materials.push(Material::new(
            device,
            &m.name,
            diffuse_texture,
            normal_texture,
            layout,
        ));
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let mut vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                    tangent: [0.0; 3],
                    bitangent: [0.0; 3],
                })
                .collect::<Vec<_>>();

            let indices = &m.mesh.indices;
            let mut triangles_included = vec![0; vertices.len()];

            for c in indices.chunks(3) {
                let v0 = vertices[c[0] as usize];
                let v1 = vertices[c[1] as usize];
                let v2 = vertices[c[2] as usize];

                let pos0: Vector3<_> = v0.position.into();
                let pos1: Vector3<_> = v1.position.into();
                let pos2: Vector3<_> = v2.position.into();

                let uv0: Vector2<_> = v0.tex_coords.into();
                let uv1: Vector2<_> = v1.tex_coords.into();
                let uv2: Vector2<_> = v2.tex_coords.into();

                // Calculate the edges of the triangle
                let delta_pos1 = pos1 - pos0;
                let delta_pos2 = pos2 - pos0;

                // This will give us a direction to calculate the
                // tangent and bitangent
                let delta_uv1 = uv1 - uv0;
                let delta_uv2 = uv2 - uv0;

                // Solving the following system of equations will
                // give us the tangent and bitangent.
                //     delta_pos1 = delta_uv1.x * T + delta_u.y * B
                //     delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
                // Luckily, the place I found this equation provided
                // the solution!
                let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                // We flip the bitangent to enable right-handed normal
                // maps with wgpu texture coordinate system
                let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * -r;

                // We'll use the same tangent/bitangent for each vertex in the triangle
                vertices[c[0] as usize].tangent =
                    (tangent + Vector3::from(vertices[c[0] as usize].tangent)).into();
                vertices[c[1] as usize].tangent =
                    (tangent + Vector3::from(vertices[c[1] as usize].tangent)).into();
                vertices[c[2] as usize].tangent =
                    (tangent + Vector3::from(vertices[c[2] as usize].tangent)).into();
                vertices[c[0] as usize].bitangent =
                    (bitangent + Vector3::from(vertices[c[0] as usize].bitangent)).into();
                vertices[c[1] as usize].bitangent =
                    (bitangent + Vector3::from(vertices[c[1] as usize].bitangent)).into();
                vertices[c[2] as usize].bitangent =
                    (bitangent + Vector3::from(vertices[c[2] as usize].bitangent)).into();

                // Used to average the tangents/bitangents
                triangles_included[c[0] as usize] += 1;
                triangles_included[c[1] as usize] += 1;
                triangles_included[c[2] as usize] += 1;
            }

            // Average the tangents/bitangents
            for (i, n) in triangles_included.into_iter().enumerate() {
                let denom = 1.0 / n as f32;
                let v = &mut vertices[i];
                v.tangent = (Vector3::from(v.tangent) * denom).into();
                v.bitangent = (Vector3::from(v.bitangent) * denom).into();
            }

            let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: cast_slice(&vertices),
                usage: BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: cast_slice(&m.mesh.indices),
                usage: BufferUsages::INDEX,
            });

            Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(Model { meshes, materials })
}

pub fn create_plane(
    device: &Device,
    queue: &Queue,
    layout: &BindGroupLayout,
    rgba: Color,
) -> Model {
    let vertices = vec![
        ModelVertex {
            position: [-1.0, 0.0, -1.0],
            tex_coords: [0.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        },
        ModelVertex {
            position: [1.0, 0.0, -1.0],
            tex_coords: [1.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        },
        ModelVertex {
            position: [1.0, 0.0, 1.0],
            tex_coords: [1.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        },
        ModelVertex {
            position: [-1.0, 0.0, 1.0],
            tex_coords: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        },
    ];

    let indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];

    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Plane Vertex Buffer"),
        contents: cast_slice(&vertices),
        usage: BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Plane Index Buffer"),
        contents: cast_slice(&indices),
        usage: BufferUsages::INDEX,
    });

    let texture = Texture::from_color(device, queue, rgba, Some("Plane texture"));
    let material = Material::new(device, "white_material", texture.clone(), texture, layout);

    let mesh = Mesh {
        name: "plane".to_string(),
        vertex_buffer,
        index_buffer,
        num_elements: indices.len() as u32,
        material: 0,
    };

    Model {
        meshes: vec![mesh],
        materials: vec![material],
    }
}

pub fn create_sphere(
    device: &Device,
    queue: &Queue,
    layout: &BindGroupLayout,
    radius: f32,
    latitude_segments: u32,
    longitude_segments: u32,
    rgba: Color,
) -> Model {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for y in 0..=latitude_segments {
        let theta = y as f32 / latitude_segments as f32 * PI;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for x in 0..=longitude_segments {
            let phi = x as f32 / longitude_segments as f32 * 2.0 * PI;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let position = [
                radius * sin_theta * cos_phi,
                radius * cos_theta,
                radius * sin_theta * sin_phi,
            ];
            let normal = [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi];
            let tex_coords = [
                x as f32 / longitude_segments as f32,
                1.0 - y as f32 / latitude_segments as f32,
            ];
            // tangent and bitangent can be approximated or calculated more accurately
            let tangent = [-sin_phi, 0.0, cos_phi];
            let bitangent = [cos_theta * cos_phi, -sin_theta, cos_theta * sin_phi];

            vertices.push(ModelVertex {
                position,
                normal,
                tex_coords,
                tangent,
                bitangent,
            });
        }
    }

    for y in 0..latitude_segments {
        for x in 0..longitude_segments {
            let a = y * (longitude_segments + 1) + x;
            let b = a + longitude_segments + 1;

            indices.push(a);
            indices.push(b);
            indices.push(a + 1);

            indices.push(b);
            indices.push(b + 1);
            indices.push(a + 1);
        }
    }

    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Sphere Vertex Buffer"),
        contents: cast_slice(&vertices),
        usage: BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Sphere Index Buffer"),
        contents: cast_slice(&indices),
        usage: BufferUsages::INDEX,
    });

    let white_texture = Texture::from_color(device, queue, rgba, Some("Sphere texture"));
    let material = Material::new(
        device,
        "white_material",
        white_texture.clone(),
        white_texture,
        layout,
    );

    let mesh = Mesh {
        name: "sphere".to_string(),
        vertex_buffer,
        index_buffer,
        num_elements: indices.len() as u32,
        material: 0,
    };

    Model {
        meshes: vec![mesh],
        materials: vec![material],
    }
}

pub fn create_cube(
    device: &Device,
    queue: &Queue,
    layout: &BindGroupLayout,
    size: f32,
    rgba: Color,
) -> Model {
    let half = size / 2.0;

    // Define 8 corners of the cube
    let positions = [
        // Front face
        [-half, -half, half],
        [half, -half, half],
        [half, half, half],
        [-half, half, half],
        // Back face
        [-half, -half, -half],
        [half, -half, -half],
        [half, half, -half],
        [-half, half, -half],
    ];

    let vertices = [
        // Front
        ModelVertex {
            position: positions[0],
            normal: [0.0, 0.0, 1.0],
            tex_coords: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[1],
            normal: [0.0, 0.0, 1.0],
            tex_coords: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[2],
            normal: [0.0, 0.0, 1.0],
            tex_coords: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[3],
            normal: [0.0, 0.0, 1.0],
            tex_coords: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        // Back
        ModelVertex {
            position: positions[5],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [0.0, 1.0],
            tangent: [-1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[4],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [1.0, 1.0],
            tangent: [-1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[7],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [1.0, 0.0],
            tangent: [-1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[6],
            normal: [0.0, 0.0, -1.0],
            tex_coords: [0.0, 0.0],
            tangent: [-1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        // Left
        ModelVertex {
            position: positions[4],
            normal: [-1.0, 0.0, 0.0],
            tex_coords: [0.0, 1.0],
            tangent: [0.0, 0.0, -1.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[0],
            normal: [-1.0, 0.0, 0.0],
            tex_coords: [1.0, 1.0],
            tangent: [0.0, 0.0, -1.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[3],
            normal: [-1.0, 0.0, 0.0],
            tex_coords: [1.0, 0.0],
            tangent: [0.0, 0.0, -1.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[7],
            normal: [-1.0, 0.0, 0.0],
            tex_coords: [0.0, 0.0],
            tangent: [0.0, 0.0, -1.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        // Right
        ModelVertex {
            position: positions[1],
            normal: [1.0, 0.0, 0.0],
            tex_coords: [0.0, 1.0],
            tangent: [0.0, 0.0, 1.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[5],
            normal: [1.0, 0.0, 0.0],
            tex_coords: [1.0, 1.0],
            tangent: [0.0, 0.0, 1.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[6],
            normal: [1.0, 0.0, 0.0],
            tex_coords: [1.0, 0.0],
            tangent: [0.0, 0.0, 1.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        ModelVertex {
            position: positions[2],
            normal: [1.0, 0.0, 0.0],
            tex_coords: [0.0, 0.0],
            tangent: [0.0, 0.0, 1.0],
            bitangent: [0.0, 1.0, 0.0],
        },
        // Top
        ModelVertex {
            position: positions[3],
            normal: [0.0, 1.0, 0.0],
            tex_coords: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: positions[2],
            normal: [0.0, 1.0, 0.0],
            tex_coords: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: positions[6],
            normal: [0.0, 1.0, 0.0],
            tex_coords: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        },
        ModelVertex {
            position: positions[7],
            normal: [0.0, 1.0, 0.0],
            tex_coords: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        },
        // Bottom
        ModelVertex {
            position: positions[4],
            normal: [0.0, -1.0, 0.0],
            tex_coords: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        },
        ModelVertex {
            position: positions[5],
            normal: [0.0, -1.0, 0.0],
            tex_coords: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        },
        ModelVertex {
            position: positions[1],
            normal: [0.0, -1.0, 0.0],
            tex_coords: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        },
        ModelVertex {
            position: positions[0],
            normal: [0.0, -1.0, 0.0],
            tex_coords: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 1.0],
        },
    ];

    let indices: Vec<u32> = (0..6)
        .flat_map(|i| {
            let base = i * 4;
            [base, base + 1, base + 2, base, base + 2, base + 3]
        })
        .collect();

    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Cube Vertex Buffer"),
        contents: cast_slice(&vertices),
        usage: BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Cube Index Buffer"),
        contents: cast_slice(&indices),
        usage: BufferUsages::INDEX,
    });

    let texture = Texture::from_color(device, queue, rgba, Some("Cube Texture"));

    let material = Material::new(device, "cube_material", texture.clone(), texture, layout);

    let mesh = Mesh {
        name: "cube".to_string(),
        vertex_buffer,
        index_buffer,
        num_elements: indices.len() as u32,
        material: 0,
    };

    Model {
        meshes: vec![mesh],
        materials: vec![material],
    }
}
