use bytemuck::{Pod, cast_slice};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};
use winit::dpi::PhysicalSize;

pub fn load_shader<'a>(shader_name: &str) -> ShaderModuleDescriptor<'a> {
    match shader_name {
        "object" => include_wgsl!("../shaders/object.wgsl"),
        "light" => include_wgsl!("../shaders/light.wgsl"),
        "hdr" => include_wgsl!("../shaders/hdr.wgsl"),
        _ => panic!("Unknown shader: {}", shader_name),
    }
}

pub fn create_pipeline_layout(
    device: &Device,
    label: &str,
    bind_group_layouts: &[&BindGroupLayout],
) -> PipelineLayout {
    device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts,
        push_constant_ranges: &[],
    })
}

pub fn create_bind_group(
    device: &Device,
    bind_group_layout: &BindGroupLayout,
    resources: &[BindingResource],
    label: &str,
) -> BindGroup {
    let entries: Vec<BindGroupEntry> = resources
        .iter()
        .enumerate()
        .map(|(i, resource)| BindGroupEntry {
            binding: i as u32,
            resource: resource.clone(),
        })
        .collect();

    device.create_bind_group(&BindGroupDescriptor {
        layout: bind_group_layout,
        entries: &entries,
        label: Some(label),
    })
}

pub fn create_buffer(
    device: &Device,
    label: &str,
    size: BufferAddress,
    usage: BufferUsages,
) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size,
        usage: usage,
        mapped_at_creation: false,
    })
}

pub fn init_buffer<T: Pod>(
    device: &Device,
    label: &str,
    buffer_contents: &[T],
    usage: BufferUsages,
) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: cast_slice(buffer_contents),
        usage: usage,
    })
}

pub fn configure_surface(
    surface_format: &TextureFormat,
    surface_capabilities: &SurfaceCapabilities,
    size: &PhysicalSize<u32>,
) -> SurfaceConfiguration {
    SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format: *surface_format,
        width: size.width,
        height: size.height,
        present_mode: surface_capabilities.present_modes[0],
        alpha_mode: surface_capabilities.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    }
}

pub fn format_surface(surface_capabilities: &SurfaceCapabilities) -> TextureFormat {
    surface_capabilities
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_capabilities.formats[0])
}

pub async fn get_device_and_queue(adapter: &Adapter) -> (Device, Queue) {
    adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features: Features::empty(),
            experimental_features: ExperimentalFeatures::disabled(),
            required_limits: if cfg!(target_arch = "wasm32") {
                Limits::downlevel_webgl2_defaults()
            } else {
                Limits::default()
            },
            memory_hints: Default::default(),
            trace: Trace::Off, // Trace path
        })
        .await
        .unwrap()
}

pub async fn create_adapter<'a>(instance: &Instance, surface: &Surface<'a>) -> Adapter {
    instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap()
}

pub fn create_wgpu_instance() -> Instance {
    Instance::new(&InstanceDescriptor {
        #[cfg(not(target_arch = "wasm32"))]
        backends: Backends::PRIMARY,
        #[cfg(target_arch = "wasm32")]
        backends: Backends::GL,
        ..Default::default()
    })
}

pub fn create_render_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    color_format: Option<TextureFormat>,
    depth_format: Option<TextureFormat>,
    vertex_layouts: &[VertexBufferLayout],
    topology: PrimitiveTopology, // NEW!
    shader: ShaderModuleDescriptor,
) -> RenderPipeline {
    let shader = device.create_shader_module(shader);

    let fragment = if color_format.is_some() {
        Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(ColorTargetState {
                format: color_format.expect("is some check failed"),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        })
    } else {
        None
    };

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(&format!("{:?}", shader)),
        layout: Some(layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: vertex_layouts,
            compilation_options: Default::default(),
        },
        fragment,
        primitive: PrimitiveState {
            topology,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: CompareFunction::LessEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}
