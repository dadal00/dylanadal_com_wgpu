// External
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::*;

pub const MAX_LIGHTS: usize = 10;
pub const MAX_LIGHT_UNIFORMS_SIZE: BufferAddress =
    (MAX_LIGHTS * size_of::<LightRaw>()) as BufferAddress;

pub fn create_lights_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(MAX_LIGHT_UNIFORMS_SIZE),
            },
            count: None,
        }],
        label: None,
    })
}

pub struct Light {
    pub pos: Vec3,
    pub color: Color,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct LightRaw {
    pub position: [f32; 3],
    pub _padding: u32,
    pub color: [f32; 3],
    pub _padding2: u32,
}

impl Light {
    pub fn to_raw(&self) -> LightRaw {
        LightRaw {
            position: [self.pos.x, self.pos.y, self.pos.z],
            _padding: 0,
            color: [
                self.color.r as f32,
                self.color.g as f32,
                self.color.b as f32,
            ],
            _padding2: 0,
        }
    }
}
