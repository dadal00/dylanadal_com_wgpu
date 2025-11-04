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
    camera::*,
    draw::*,
    hdr::HdrPipeline,
    light::*,
    models::*,
    resources::{create_plane, create_sphere},
    texture::{Texture, create_texture_bind_group_layout},
    utils::*,
};

pub struct State {
    pub window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    surface_config: SurfaceConfiguration,

    main_render_pipeline: RenderPipeline,
    plane_model: Model,

    light_render_pipeline: RenderPipeline,
    light_model: Model,

    camera: Camera,
    projection: Projection,
    pub camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,

    planes: Vec<_Instance>,
    planes_buffer: Buffer,

    light_instances: Vec<_Instance>,
    light_instance_buffer: Buffer,

    depth_texture: Texture,
    is_surface_configured: bool,

    lights: Vec<Light>,
    light_buffer: Buffer,
    light_bind_group: BindGroup,

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
        let texture_bind_group_layout = create_texture_bind_group_layout(&device);

        let surface_capabilities = surface.get_capabilities(&adapter);
        let surface_format = format_surface(&surface_capabilities);
        let surface_config = configure_surface(&surface_format, &surface_capabilities, &size);

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

        let planes = vec![_Instance {
            position: Vector3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: Quaternion::one(),
            scale: Vector3::new(100.0, 1.0, 100.0),
        }];

        let plane_data = _Instance::to_raw_vec(&planes);
        let planes_buffer = init_buffer(
            &device,
            "Instance Buffer",
            &plane_data,
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

        let plane_model = create_plane(
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

        let light_instances: Vec<_Instance> = lights
            .iter()
            .map(|light| _Instance {
                position: Vector3 {
                    x: light.pos.x,
                    y: light.pos.y,
                    z: light.pos.z,
                },
                rotation: Quaternion::one(),
                scale: Vector3::new(1.0, 1.0, 1.0),
            })
            .collect();

        let light_instance_data = _Instance::to_raw_vec(&light_instances);

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

        let main_render_pipeline = create_render_pipeline(
            &device,
            &create_pipeline_layout(
                &device,
                "Render Pipeline Layout",
                &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                ],
            ),
            Some(hdr.format()),
            Some(Texture::DEPTH_FORMAT),
            &[ModelVertex::desc(), InstanceRaw::desc()],
            PrimitiveTopology::TriangleList,
            load_shader("object"),
        );

        let light_render_pipeline = create_render_pipeline(
            &device,
            &create_pipeline_layout(
                &device,
                "Light Pipeline Layout",
                &[&texture_bind_group_layout, &camera_bind_group_layout],
            ),
            Some(hdr.format()),
            Some(Texture::DEPTH_FORMAT),
            &[ModelVertex::desc(), InstanceRaw::desc()],
            PrimitiveTopology::TriangleList,
            load_shader("light"),
        );

        Ok(Self {
            window,
            surface,
            device,
            queue,
            surface_config,

            main_render_pipeline,
            light_model,
            plane_model,

            camera,
            projection,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,

            planes,
            planes_buffer,

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

    pub fn update(&mut self, delta_time: Duration) {
        self.update_camera(delta_time);
        self.update_lights(delta_time);
    }

    pub fn render(&mut self) -> Result<(), SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let (surface_texture, texture_view) = self.get_texture_and_view()?;
        let mut command_encoder = self.get_encoder();

        {
            let mut render_pass = self.create_main_render_pass(&mut command_encoder);

            self.render_lights(&mut render_pass);
            self.render_plane(&mut render_pass);
        }

        self.hdr.process(&mut command_encoder, &texture_view);
        self.queue.submit(once(command_encoder.finish()));
        surface_texture.present();

        Ok(())
    }

    fn render_plane<'b: 'a, 'a>(&'b self, render_pass: &mut RenderPass<'a>) {
        self.render_objects(
            render_pass,
            &self.planes_buffer,
            &self.main_render_pipeline,
            &self.plane_model,
            self.planes.len() as u32,
            &self.camera_bind_group,
            Some(&self.light_bind_group),
        );
    }

    fn render_lights<'this: 'render, 'render>(&'this self, render_pass: &mut RenderPass<'render>) {
        self.render_objects(
            render_pass,
            &self.light_instance_buffer,
            &self.light_render_pipeline,
            &self.light_model,
            self.light_instances.len() as u32,
            &self.camera_bind_group,
            None,
        );
    }

    fn render_objects<'b: 'a, 'a>(
        &'b self,
        render_pass: &mut RenderPass<'a>,
        instance_buffer: &Buffer,
        render_pipeline: &RenderPipeline,
        model: &'a Model,
        num_instances: u32,
        camera_bind_group: &'a BindGroup,
        light_bind_group: Option<&'a BindGroup>,
    ) {
        render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        render_pass.set_pipeline(render_pipeline);
        render_pass.draw_model_instanced(
            model,
            0..num_instances,
            camera_bind_group,
            light_bind_group,
        );
    }

    fn get_encoder(&self) -> CommandEncoder {
        self.device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Main Render Encoder"),
            })
    }

    fn get_texture_and_view(&self) -> Result<((SurfaceTexture, TextureView)), SurfaceError> {
        let surface_texture = self.surface.get_current_texture()?;
        let texture_view = surface_texture
            .texture
            .create_view(&TextureViewDescriptor::default());

        Ok((surface_texture, texture_view))
    }

    fn update_camera(&mut self, delta_time: Duration) {
        self.camera_controller
            .update_camera(&mut self.camera, delta_time);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);

        self.queue
            .write_buffer(&self.camera_buffer, 0, cast_slice(&[self.camera_uniform]));
    }

    fn update_lights(&mut self, delta_time: Duration) {
        for (i, light) in self.lights.iter_mut().enumerate() {
            let old_pos: Vector3<f32> = [light.pos.x, light.pos.y, light.pos.z].into();

            let rotation =
                Quaternion::from_axis_angle(Vector3::unit_y(), Deg(PI * delta_time.as_secs_f32()));

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

    fn create_main_render_pass<'a>(
        &self,
        command_encoder: &'a mut CommandEncoder,
    ) -> RenderPass<'a> {
        command_encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Main Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: self.hdr.view(),
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.7,
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
        })
    }
}
