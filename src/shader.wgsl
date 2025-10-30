// Vertex shader

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(1) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
    proj: mat4x4<f32>,
}
@group(2) @binding(0)
var<uniform> lights: array<Light, 10>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
}
struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) tangent_position: vec3<f32>,
    @location(2) tangent_light_0: vec3<f32>,
    @location(3) tangent_light_1: vec3<f32>,
    @location(4) tangent_view_position: vec3<f32>,
    @location(5) world_position: vec4<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    // Construct the tangent matrix
    let world_normal = normalize(normal_matrix * model.normal);
    let world_tangent = normalize(normal_matrix * model.tangent);
    let world_bitangent = normalize(normal_matrix * model.bitangent);
    let tangent_matrix = transpose(mat3x3<f32>(
        world_tangent,
        world_bitangent,
        world_normal,
    ));

    let world_position = model_matrix * vec4<f32>(model.position, 1.0);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_position;
    out.tex_coords = model.tex_coords;
    out.tangent_position = tangent_matrix * world_position.xyz;
    out.tangent_view_position = tangent_matrix * camera.view_pos.xyz;

    out.tangent_light_0 = tangent_matrix * lights[0].position;
    out.tangent_light_1 = tangent_matrix * lights[1].position;

    out.world_position = world_position;

    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0)@binding(1)
var s_diffuse: sampler;
@group(0)@binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;

struct ShadowUniform {
    proj: mat4x4<f32>,
};
@group(2) @binding(1)
var<uniform> shadow_uniform: ShadowUniform;
@group(3) @binding(0)
var t_shadow: texture_depth_2d_array;
@group(3) @binding(1)
var sampler_shadow: sampler_comparison;

fn fetch_shadow(light_id: u32, homogeneous_coords: vec4<f32>) -> f32 {
    if (homogeneous_coords.w <= 0.0) {
        return 1.0;
    }

    let ndc = homogeneous_coords.xyz / homogeneous_coords.w;                     // -1..1 range
    let uv = vec2<f32>(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y); // Convert to 0..1 and flip Y
    let depth = 0.5 * ndc.z + 0.5;                 // Convert z to 0..1

    return textureSampleCompareLevel(t_shadow, sampler_shadow, uv, i32(light_id), depth);
    // // compensate for the Y-flip difference between the NDC and texture coordinates
    // let flip_correction = vec2<f32>(0.5, -0.5);
    // // compute texture coordinates for shadow lookup
    // let proj_correction = 1.0 / homogeneous_coords.w;
    // let light_local = homogeneous_coords.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
    // // do the lookup, using HW PCF and comparison
    // return textureSampleCompareLevel(t_shadow, sampler_shadow, light_local, i32(light_id), homogeneous_coords.z * proj_correction);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let object_normal: vec4<f32> = textureSample(t_normal, s_normal, in.tex_coords);
    
    let tangent_normal = object_normal.xyz * 2.0 - 1.0;



    // let light_vec = in.tangent_light_1 - in.tangent_position;
    // let light_distance = length(light_vec);
    // let light_dir = normalize(light_vec);

    // let attenuation = 1.0 / (1.0 + 0.09 * light_distance + 0.032 * light_distance * light_distance);
    // let focused_attenuation = pow(attenuation, 2.0);

    // let diffuse_color = lights[1].color * focused_attenuation;

    // let shadow = fetch_shadow(1, lights[1].proj * in.world_position);




    let light_vec1 = in.tangent_light_0 - in.tangent_position;
    let light_distance1 = length(light_vec1);
    let light_dir1 = normalize(light_vec1);

    let attenuation1 = 1.0 / (1.0 + 0.09 * light_distance1 + 0.032 * light_distance1 * light_distance1);
    let focused_attenuation1 = pow(attenuation1, 2.0);

    let diffuse_color1 = lights[0].color * focused_attenuation1;
    let shadow1 = fetch_shadow(0, lights[0].proj * in.world_position);



    let result =  shadow1 * diffuse_color1 * object_color.xyz;

    return vec4<f32>(result, object_color.a);
}
