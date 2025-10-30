struct Camera {
    view_position: vec4<f32>,
    view_projection: mat4x4<f32>,
    number_of_lights: vec4<u32>,
}
@group(1) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}
@group(2) @binding(0)
var<uniform> lights: array<Light, 10>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) texture_coordinates: vec2<f32>,
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
    @location(0) texture_coordinates: vec2<f32>,
    
    @location(1) tangent_position: vec3<f32>,
    @location(2) tangent_light_0: vec3<f32>,
    @location(3) tangent_light_1: vec3<f32>,
    @location(4) tangent_view_position: vec3<f32>,

    @location(5) tangent_matrix_row_0: vec3<f32>,
    @location(6) tangent_matrix_row_1: vec3<f32>,
    @location(7) tangent_matrix_row_2: vec3<f32>,
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
    out.clip_position = camera.view_projection * world_position;
    out.texture_coordinates = model.texture_coordinates;
    out.tangent_position = tangent_matrix * world_position.xyz;
    out.tangent_view_position = tangent_matrix * camera.view_position.xyz;

    out.tangent_matrix_row_0 = tangent_matrix[0];
    out.tangent_matrix_row_1 = tangent_matrix[1];
    out.tangent_matrix_row_2 = tangent_matrix[2];

    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0)@binding(1)
var s_diffuse: sampler;
@group(0)@binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tangent_matrix = mat3x3<f32>(
        in.tangent_matrix_row_0,
        in.tangent_matrix_row_1,
        in.tangent_matrix_row_2,
    );

    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.texture_coordinates);
    let object_normal: vec4<f32> = textureSample(t_normal, s_normal, in.texture_coordinates);
    let tangent_normal = object_normal.xyz * 2.0 - 1.0;

    var result: vec3<f32> = vec3<f32>(0.0);

    for (var i = 0u; i < camera.number_of_lights[0]; i = i + 1u){
        let light_vec = tangent_matrix * lights[i].position - in.tangent_position;
        let light_distance = length(light_vec);
        let light_dir = normalize(light_vec);

        let attenuation = 1.0 / (1.0 + 0.09 * light_distance + 0.032 * light_distance * light_distance);
        let focused_attenuation = pow(attenuation, 2.0);
        let diffuse_color = lights[i].color * focused_attenuation;

        result += diffuse_color * object_color.xyz;
    }

    return vec4<f32>(result, object_color.a);
}
