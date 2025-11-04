// External
use wgpu::*;

// Internal Modules
use crate::{
    texture::Texture,
    utils::{create_bind_group, create_pipeline_layout, create_render_pipeline, load_shader},
};

pub struct HdrPipeline {
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    texture: Texture,
    width: u32,
    height: u32,
    format: TextureFormat,
    layout: BindGroupLayout,
}

impl HdrPipeline {
    pub fn new(device: &Device, config: &SurfaceConfiguration) -> Self {
        // Odd bug where hdr breaks if at some point width + height 0
        let width = config.width.max(1);
        let height = config.height.max(1);

        // We could use `Rgba32Float`, but that requires some extra
        // features to be enabled.
        let format = TextureFormat::Rgba16Float;

        log::info!("Creating HDR texture: {}x{}", width, height);

        let texture = Texture::create_2d_texture(
            device,
            width,
            height,
            format,
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            FilterMode::Nearest,
            Some("Hdr::texture"),
        );

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Hdr::layout"),
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
        });
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
            format,
        }
    }

    /// Resize the HDR texture
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        self.texture = Texture::create_2d_texture(
            device,
            width,
            height,
            self.format,
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

    /// Exposes the HDR texture
    pub fn view(&self) -> &TextureView {
        &self.texture.view
    }

    /// The format of the HDR texture
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// This renders the internal HDR texture to the [TextureView]
    /// supplied as parameter.
    pub fn process(&self, encoder: &mut CommandEncoder, output: &TextureView) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Hdr::process"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &output,
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
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
