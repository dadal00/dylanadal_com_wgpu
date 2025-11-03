// External
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::Color;

pub const MAX_LIGHTS: usize = 10;

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
