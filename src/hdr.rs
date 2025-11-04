// External
use wgpu::*;

// Internal Modules
use crate::{texture::Texture, utils::*};

const HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

pub struct HdrPipeline {
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    texture: Texture,
    width: u32,
    height: u32,
    layout: BindGroupLayout,
}

impl HdrPipeline {
    pub fn new(device: &Device, config: &SurfaceConfiguration) -> Self {
        // Odd bug where HDR breaks if at some point width + height 0
        let width = config.width.max(1);
        let height = config.height.max(1);

        let texture = Texture::create_2d_texture(
            device,
            width,
            height,
            HDR_FORMAT,
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            FilterMode::Nearest,
            Some("Hdr::texture"),
        );

        let layout = create_hdr_bind_group_layout(&device, "HDR Bind Group Layout");

        let bind_group = create_bind_group(
            &device,
            &layout,
            &[
                BindingResource::TextureView(&texture.view),
                BindingResource::Sampler(&texture.sampler),
            ],
            "HDR Bind Group",
        );

        let pipeline = create_render_pipeline(
            device,
            &create_pipeline_layout(&device, "HDR Pipeline Layout", &[&layout]),
            Some(config.format.add_srgb_suffix()),
            None,
            &[],
            PrimitiveTopology::TriangleList,
            load_shader("hdr"),
        );

        Self {
            pipeline,
            bind_group,
            layout,
            texture,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        self.texture = Texture::create_2d_texture(
            device,
            width,
            height,
            HDR_FORMAT,
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            FilterMode::Nearest,
            Some("Hdr::texture"),
        );

        self.bind_group = create_bind_group(
            &device,
            &self.layout,
            &[
                BindingResource::TextureView(&self.texture.view),
                BindingResource::Sampler(&self.texture.sampler),
            ],
            "HDR Bind Group",
        );

        self.width = width;
        self.height = height;
    }

    pub fn view(&self) -> &TextureView {
        &self.texture.view
    }

    pub fn format(&self) -> TextureFormat {
        HDR_FORMAT
    }

    pub fn process(&self, command_encoder: &mut CommandEncoder, texture_view: &TextureView) {
        let mut hdr_render_pass =
            create_hdr_render_pass(command_encoder, "HDR Render Pass", texture_view);

        self.render_hdr(&mut hdr_render_pass);
    }

    fn render_hdr(&self, hdr_render_pass: &mut RenderPass) {
        hdr_render_pass.set_pipeline(&self.pipeline);
        hdr_render_pass.set_bind_group(0, &self.bind_group, &[]);
        hdr_render_pass.draw(0..3, 0..1);
    }
}

fn create_hdr_render_pass<'a>(
    command_encoder: &'a mut CommandEncoder,
    label: &str,
    texture_view: &TextureView,
) -> RenderPass<'a> {
    command_encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: texture_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
    })
}

fn create_hdr_bind_group_layout(device: &Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    // The Rgba16Float format cannot be filtered
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}
