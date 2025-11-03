// Standard Library
use std::{f32::consts::PI, iter::once, sync::Arc, time::Duration};

// External
use anyhow::Result;
use bytemuck::cast_slice;
use cgmath::{Deg, Quaternion, Vector3, prelude::*};
use glam::Vec3;
use wgpu::*;
use winit::{event::*, event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};

// Internal Modules
use crate::{
    camera::{
        Camera, CameraController, CameraUniform, Projection, create_camera_bind_group_layout,
    },
    hdr::HdrPipeline,
    light::{Light, LightRaw, MAX_LIGHT_UNIFORMS_SIZE, create_lights_bind_group_layout},
    model::{DrawLight, DrawModel, Instance, InstanceRaw, Model, ModelVertex, Vertex},
    resources::{create_plane, create_sphere},
    texture::{Texture, create_texture_bind_group_layout},
    utils::{
        configure_surface, create_adapter, create_bind_group, create_buffer,
        create_render_pipeline, create_wgpu_instance, format_surface, get_device_and_queue,
        init_buffer,
    },
};

pub struct State {
    pub window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    surface_config: SurfaceConfiguration,

    render_pipeline: RenderPipeline,
    light_model: Model,
    obj_model: Model,

    camera: Camera,
    projection: Projection,
    pub camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,

    instances: Vec<Instance>,
    instance_buffer: Buffer,

    light_instances: Vec<Instance>,
    light_instance_buffer: Buffer,

    depth_texture: Texture,
    is_surface_configured: bool,

    lights: Vec<Light>,
    light_buffer: Buffer,
    light_bind_group: BindGroup,
    light_render_pipeline: RenderPipeline,

    pub mouse_pressed: bool,
    hdr: HdrPipeline,
}

impl State {
    pub async fn new(window: Arc<Window>) -> Result<State> {
        let size = window.inner_size();

        let instance = create_wgpu_instance();

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = create_adapter(&instance, &surface).await;
        let (device, queue) = get_device_and_queue(&adapter).await;

        let surface_capabilities = surface.get_capabilities(&adapter);

        let surface_format = format_surface(&surface_capabilities);
        let surface_config = configure_surface(&surface_format, &surface_capabilities, &size);

        let texture_bind_group_layout = create_texture_bind_group_layout(&device);

        let camera = Camera::new((0.0, 5.0, 10.0), Deg(-90.0), Deg(-20.0));
        let projection = Projection::new(
            surface_config.width,
            surface_config.height,
            Deg(45.0),
            0.1,
            100.0,
        );
        let camera_controller = CameraController::new(4.0, 0.4);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let lights = vec![
            Light {
                pos: Vec3::new(3.0, 2.0, 3.0),
                color: Color {
                    r: 1.0,
                    g: 0.5,
                    b: 0.5,
                    a: 1.0,
                },
            },
            Light {
                pos: Vec3::new(-3.0, 2.0, -3.0),
                color: Color {
                    r: 1.0,
                    g: 0.5,
                    b: 0.5,
                    a: 1.0,
                },
            },
        ];

        camera_uniform.set_lights(lights.len().try_into().unwrap());

        let camera_buffer = init_buffer(
            &device,
            "Camera Buffer",
            &[camera_uniform],
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let instances = vec![Instance {
            position: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: Quaternion::one(),
            scale: Vector3::new(100.0, 1.0, 100.0),
        }];

        let instance_data = Instance::to_raw_vec(&instances);
        let instance_buffer = init_buffer(
            &device,
            "Instance Buffer",
            &instance_data,
            BufferUsages::VERTEX,
        );

        let camera_bind_group_layout = create_camera_bind_group_layout(&device);

        let camera_bind_group = create_bind_group(
            &device,
            &camera_bind_group_layout,
            &[camera_buffer.as_entire_binding()],
            "Camera Bind Group",
        );

        let light_model = create_sphere(
            &device,
            &queue,
            &texture_bind_group_layout,
            1.0,
            32,
            32,
            lights[1].color,
        );

        let obj_model = create_plane(
            &device,
            &queue,
            &texture_bind_group_layout,
            Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        );

        let light_instances: Vec<Instance> = lights
            .iter()
            .map(|light| Instance {
                position: Vector3 {
                    x: light.pos.x,
                    y: light.pos.y,
                    z: light.pos.z,
                },
                rotation: Quaternion::one(),
                scale: Vector3::new(1.0, 1.0, 1.0),
            })
            .collect();

        let light_instance_data = Instance::to_raw_vec(&light_instances);

        let light_instance_buffer = init_buffer(
            &device,
            "Light Instance Buffer",
            &light_instance_data,
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
        );

        let light_buffer = create_buffer(
            &device,
            "Light Buffer",
            MAX_LIGHT_UNIFORMS_SIZE,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let light_bind_group_layout = create_lights_bind_group_layout(&device);

        let light_bind_group = create_bind_group(
            &device,
            &light_bind_group_layout,
            &[light_buffer.as_entire_binding()],
            "Light Buffer",
        );

        let depth_texture = Texture::create_depth_texture(
            &device,
            surface_config.width,
            surface_config.height,
            1,
            "depth_texture",
        );

        let hdr = HdrPipeline::new(&device, &surface_config);

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &camera_bind_group_layout,
                &light_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = {
            let shader = ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                Some(hdr.format()),
                Some(Texture::DEPTH_FORMAT),
                &[ModelVertex::desc(), InstanceRaw::desc()],
                PrimitiveTopology::TriangleList,
                shader,
            )
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                Some(hdr.format()),
                Some(Texture::DEPTH_FORMAT),
                &[ModelVertex::desc(), InstanceRaw::desc()],
                PrimitiveTopology::TriangleList,
                shader,
            )
        };

        Ok(Self {
            window,
            surface,
            device,
            queue,
            surface_config,

            render_pipeline,
            light_model,
            obj_model,

            camera,
            projection,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,

            instances,
            instance_buffer,

            light_instances,
            light_instance_buffer,

            depth_texture,
            is_surface_configured: false,

            lights,
            light_buffer,
            light_bind_group,
            light_render_pipeline,

            mouse_pressed: false,
            hdr,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.projection.resize(width, height);
            self.hdr.resize(&self.device, width, height);

            self.surface_config.width = width;
            self.surface_config.height = height;

            self.surface.configure(&self.device, &self.surface_config);
            self.is_surface_configured = true;

            self.depth_texture = Texture::create_depth_texture(
                &self.device,
                self.surface_config.width,
                self.surface_config.height,
                1,
                "depth_texture",
            );
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        if !self.camera_controller.handle_key(key, pressed) {
            match (key, pressed) {
                (KeyCode::Escape, true) => event_loop.exit(),
                _ => {}
            }
        }
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.mouse_pressed = pressed,
            _ => {}
        }
    }

    pub fn handle_mouse_scroll(&mut self, delta: &MouseScrollDelta) {
        self.camera_controller.handle_scroll(delta);
    }

    pub fn update(&mut self, dt: Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);

        self.queue
            .write_buffer(&self.camera_buffer, 0, cast_slice(&[self.camera_uniform]));

        for (i, light) in self.lights.iter_mut().enumerate() {
            let old_pos: Vector3<f32> = [light.pos.x, light.pos.y, light.pos.z].into();

            let rotation =
                Quaternion::from_axis_angle(Vector3::unit_y(), Deg(PI * dt.as_secs_f32()));

            let new_pos = rotation * old_pos;

            light.pos = [new_pos[0], new_pos[1], new_pos[2]].into();
            self.light_instances[i].position = Vector3 {
                x: light.pos.x,
                y: light.pos.y,
                z: light.pos.z,
            };

            self.queue.write_buffer(
                &self.light_buffer,
                (i * size_of::<LightRaw>()) as BufferAddress,
                cast_slice(&[light.to_raw()]),
            );

            self.queue.write_buffer(
                &self.light_instance_buffer,
                (i * size_of::<InstanceRaw>()) as BufferAddress,
                cast_slice(&[self.light_instances[i].to_raw()]),
            );
        }
    }

    pub fn render(&mut self) -> Result<(), SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: self.hdr.view(),
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_vertex_buffer(1, self.light_instance_buffer.slice(..));
            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_model_instanced(
                &self.light_model,
                0..self.light_instances.len() as u32,
                &self.camera_bind_group,
            );

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.obj_model,
                0..self.instances.len() as u32,
                &self.camera_bind_group,
                &self.light_bind_group,
            );
        }
        self.hdr.process(&mut encoder, &view);

        self.queue.submit(once(encoder.finish()));
        output.present();

        Ok(())
    }
}
