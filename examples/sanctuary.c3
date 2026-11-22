// "The Light Sanctuary" — flagship showcase of drie's lighting, GI, materials, and post stack:
//
//   - Split Radiance Cascades (sparse GPU GI) vs Dense RC vs Off (Tab / 1-3)
//   - Froxel Volumetric Fog god rays + VSM soft shadows slicing through a concrete colonnade (V)
//   - Screen-Space Reflections (SSR with back-face thickness) on wet marble + Planar Reflector mirror (R)
//   - Paired LTC RectAreaLights + emissive monoliths (sharp speculars + soft GI diffuse bounce)
//   - Hero materials: dispersive transmissive crystal orb, brushed anisotropic gold ring,
//     iridescent clearcoat sphere, and a live animated drie::nodes molten-veined torus
//   - Full HDR post stack: GTAO (G) + UnrealBloom (B) + TAA + ACES Filmic (--hdr / --p3 supported)
//   - Live F3 probe debugger (drie::rc_debug)
//
//   just example sanctuary                       open interactive window (--hdr, --p3)
//   just example sanctuary --png out.png         render headless still (--mode split|dense|off)
module sanctuary_example;
import drie, drie::nodes, drie::postprocessing, drie::ui, drie::controls, drie::rc_debug, drie::readback, drie::png, sokol;
import std::io, std::math;

const int W = 1280;
const int H = 720;

enum RcMode : int
{
    SPLIT,
    DENSE,
    OFF,
}

const String[3] MODE_NAMES = { "SPLIT RC", "DENSE RC", "OFF" };

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
OrbitControls* controls;
RenderTarget* scene_rt;
RenderTarget* back_rt;
GTAOPass* ao;
UnrealBloomPass* bloom;
TAAPass* taa;
OutputPass* output;
drie::Mat4 prev_view_proj;
bool has_prev_vp = false;

RadianceCascades* dense_rc;
RadianceCascades* split_rc;
Fog* vol_fog;
Reflector* wall_mirror;
MeshStandardMaterial* floor_mat;
MeshStandardMaterial* mirror_mat;
SpotLight* key_spot;
SpotLight* rim_spot;
SpotLight* apse_spot;
Mesh* crystal_orb;
Mesh* gold_halo;
Mesh* node_torus;
Node pulse_u;

RcMode mode = SPLIT;
bool sweep = true;
bool lights_on = true;
bool fog_on = true;
bool refl_on = true;
bool gtao_on = true;
bool bloom_on = true;

double sim_time = 0.0;
double last_frame_t = -1.0;
double smooth_ms = 16.0;
ulong frame_count = 1;
ulong last_tab_frame = 0;

// ---- procedural drie::nodes shader for the animated molten-veined obsidian torus

fn Node hash3(Node cell)
{
    Node v = nodes::convert(nodes::convert(cell, IVEC3), UVEC3) * nodes::uint_(1664525) + nodes::uint_(1013904223);
    Node x = v.x() + v.y() * v.z();
    Node y = v.y() + v.z() * x;
    Node z = v.z() + x * y;
    Node h = nodes::bit_xor(x + y * z, nodes::shift_right(z, nodes::uint_(16)));
    h = h * nodes::uint_(0x27d4eb2d);
    h = nodes::bit_xor(h, nodes::shift_right(h, nodes::uint_(15)));
    return nodes::convert(h, FLOAT) * (1.0f / 4294967296.0f);
}

fn Node value_noise3(Node p)
{
    Node i = nodes::floor(p);
    Node f = nodes::fract(p);
    Node u = f * f * (3.0f - f * 2.0f);
    Node[8] c;
    for (int k = 0; k < 8; k++) c[k] = hash3(i + nodes::vec3(k & 1, (k >> 1) & 1, k >> 2));
    Node x0 = nodes::mix(nodes::mix(c[0], c[1], u.x()), nodes::mix(c[2], c[3], u.x()), u.y());
    Node x1 = nodes::mix(nodes::mix(c[4], c[5], u.x()), nodes::mix(c[6], c[7], u.x()), u.y());
    return nodes::mix(x0, x1, u.z());
}

fn Node fbm3(Node p)
{
    Node sum = nodes::variable(nodes::num(0));
    Node amp = nodes::variable(nodes::num(0.5f));
    Node q = nodes::variable(p);
    Node i = nodes::loop_index();
    return nodes::seq(nodes::block(
        sum, amp, q,
        nodes::loop(i, 0, 4, nodes::block(
            sum.assign(sum + value_noise3(q) * amp),
            q.assign(q * 2.04f + 1.37f),
            amp.assign(amp * 0.5f)))),
        sum);
}

fn Material* molten_node_material()
{
    Material* m = drie::mesh_standard_material(color: 0xffffff, roughness: 0.32f, metalness: 0.78f);
    pulse_u = nodes::uniform(1.0f);
    Node p = nodes::position_local();
    Node t = nodes::time();
    Node n = fbm3(p * 3.8f + nodes::vec3(0.0f, t * 0.25f, 0.0f));
    // Ridged molten veins cutting across dark hammered obsidian-bronze
    Node ridge = nodes::one_minus(nodes::abs(n * 2.0f - 1.0f));
    Node vein = nodes::pow(nodes::smoothstep(0.76f, 0.97f, ridge), 3.2f);
    m.color_node = nodes::mix(nodes::color(0x181a22), nodes::color(0x6e442c), nodes::smoothstep(0.35f, 0.72f, n));
    m.emissive_node = nodes::mix(nodes::color(0xff3008), nodes::color(0xffdc80), vein) * (vein * 5.5f * pulse_u);
    // Subtle breathing organic ripple along the torus
    Node swell = 1.0f + nodes::sin(p.y() * 5.0f - t * 2.0f) * 0.04f;
    m.position_node = p * swell;
    return m;
}

// Iridescence thickness ramp texture (180..720 nm)
fn Texture* iridescence_ramp()
{
    int w = 4;
    int h = 64;
    char[] px = mem::new_array(char, (usz)(w * h * 4));
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            usz i = (usz)((y * w + x) * 4);
            px[i + 1] = (char)(255 * y / (h - 1));
            px[i + 3] = 255;
        }
    }
    return drie::data_texture(px, w, h, NO_COLOR_SPACE);
}

fn Mesh* add_box(Vector3 pos, Vector3 size, Material* mat, bool cast_shadow = true, bool recv_shadow = true)
{
    Mesh* m = drie::mesh(drie::box_geometry(size.x, size.y, size.z), mat);
    m.position = pos;
    m.cast_shadow = cast_shadow;
    m.receive_shadow = recv_shadow;
    scene.add(m);
    return m;
}

fn void build_scene()
{
    scene = drie::scene();
    scene.background = 0x05070c;

    // Volumetric froxel fog for crisp, atmospheric spotlight shafts through the colonnade
    vol_fog = drie::volumetric_fog(0x07090f, density: 0.015f, light_scatter: 0.0014f);
    scene.fog = vol_fog;

    // ---- architectural materials
    floor_mat = drie::mesh_standard_material(color: 0x20242c, roughness: 0.10f, metalness: 0.10f);
    floor_mat.ssr = 1;

    MeshStandardMaterial* concrete = drie::mesh_standard_material(color: 0x9c9890, roughness: 0.88f);
    MeshStandardMaterial* dark_slate = drie::mesh_standard_material(color: 0x2a2d34, roughness: 0.20f, metalness: 0.15f);
    dark_slate.ssr = 1;
    MeshStandardMaterial* brass = drie::mesh_standard_material(color: 0xd4a458, roughness: 0.24f, metalness: 1.0f);
    brass.anisotropy = 0.85f;

    MeshStandardMaterial* vermilion = drie::mesh_standard_material(color: 0xdc3020, roughness: 0.88f);
    MeshStandardMaterial* cobalt = drie::mesh_standard_material(color: 0x1e56e8, roughness: 0.88f);
    MeshStandardMaterial* saffron = drie::mesh_standard_material(color: 0xe8a428, roughness: 0.86f);

    MeshStandardMaterial* cyan_emissive = drie::mesh_standard_material(
        color: 0x000000, roughness: 0.9f, emissive: 0x18e8ff, emissive_intensity: 14.0f);
    MeshStandardMaterial* amber_emissive = drie::mesh_standard_material(
        color: 0x000000, roughness: 0.9f, emissive: 0xff881c, emissive_intensity: 14.0f);
    MeshStandardMaterial* magenta_emissive = drie::mesh_standard_material(
        color: 0x000000, roughness: 0.9f, emissive: 0xff2888, emissive_intensity: 9.5f);

    // ---- brutalist gallery shell: X in [-5.0, 5.0], Y in [0.0, 4.4], Z in [-6.0, 5.0]
    Mesh* floor = drie::mesh(drie::plane_geometry(10.4f, 11.4f), floor_mat);
    floor.rotation.x = -drie::PI / 2;
    floor.position = { 0, 0, -0.5f };
    floor.receive_shadow = true;
    scene.add(floor);

    add_box({ 0, 4.5f, -0.5f }, { 10.4f, 0.2f, 11.4f }, concrete, false, true);
    add_box({ 0, 2.2f, -6.1f }, { 10.4f, 4.6f, 0.2f }, concrete, true, true);
    add_box({ 0, 2.2f, 5.1f }, { 10.4f, 4.6f, 0.2f }, concrete, false, true);

    // Vivid vermilion left baffle & cobalt right baffle for rich Split RC color bleeding
    add_box({ -5.1f, 2.2f, -0.5f }, { 0.2f, 4.6f, 11.4f }, vermilion, true, true);
    add_box({ 5.1f, 2.2f, -0.5f }, { 0.2f, 4.6f, 11.4f }, cobalt, true, true);

    // Coffered concrete ceiling ribs (catch bounced RC light and carve volumetric beams)
    for (int i = 0; i < 4; i++)
    {
        float z = -4.2f + (float)i * 2.4f;
        add_box({ 0, 4.05f, z }, { 10.0f, 0.7f, 0.38f }, concrete, true, true);
    }

    // Colonnade: 4 brutalist concrete pillars creating dramatic volumetric light shafts
    float[2] col_x = { -3.05f, 3.05f };
    float[2] col_z = { -0.8f, -3.1f };
    foreach (cx : col_x)
    {
        foreach (cz : col_z)
        {
            add_box({ cx, 2.15f, cz }, { 0.56f, 4.3f, 0.56f }, concrete, true, true);
            add_box({ cx, 0.14f, cz }, { 0.72f, 0.28f, 0.72f }, dark_slate, true, true);
        }
    }

    // Monumental archway framing the rear saffron apse (Z = -4.0)
    add_box({ -3.35f, 2.2f, -4.0f }, { 3.3f, 4.4f, 0.34f }, concrete, true, true);
    add_box({ 3.35f, 2.2f, -4.0f }, { 3.3f, 4.4f, 0.34f }, concrete, true, true);
    add_box({ 0.0f, 3.65f, -4.0f }, { 3.6f, 1.5f, 0.34f }, concrete, true, true);

    // Rear saffron-gold reflector wall inside the apse + dark architectural fins so the
    // dispersive crystal orb visibly bends high-contrast stripes into rainbow prism fringes
    add_box({ 0.0f, 2.0f, -5.85f }, { 5.2f, 4.0f, 0.16f }, saffron, true, true);
    for (int i = -2; i <= 2; i++)
    {
        add_box({ (float)i * 0.55f, 2.0f, -5.68f }, { 0.12f, 3.8f, 0.20f }, dark_slate, true, true);
    }

    // Central raised dais with brass trim & recessed magenta cove glow
    add_box({ 0.0f, 0.07f, -0.7f }, { 4.8f, 0.14f, 2.3f }, dark_slate, true, true);
    add_box({ 0.0f, 0.15f, -0.7f }, { 4.86f, 0.025f, 2.36f }, brass, false, true);
    add_box({ 0.0f, 0.03f, 0.47f }, { 4.3f, 0.05f, 0.06f }, magenta_emissive, false, false);

    // ---- Planar Reflector mirror wall on the right side (X = +4.88)
    wall_mirror = drie::reflector({ -1, 0, 0 }, -4.88f, resolution: 0.65f);
    wall_mirror.blur = 0.008f;
    wall_mirror.strength = 0.98f;
    mirror_mat = drie::mesh_standard_material(color: 0xffffff, roughness: 0.02f, metalness: 0.96f);
    mirror_mat.reflector = wall_mirror;
    // Mirror frame + surface
    add_box({ 4.94f, 1.95f, -0.3f }, { 0.10f, 3.35f, 5.8f }, brass, true, true);
    Mesh* mirror_pane = drie::mesh(drie::plane_geometry(5.6f, 3.1f), mirror_mat);
    mirror_pane.position = { 4.88f, 1.95f, -0.3f };
    mirror_pane.rotation.y = -drie::PI / 2;
    scene.add(mirror_pane);

    // ---- Pure emissive light monoliths (100% shadowed diffuse bounce via Split RC + real SSR reflections)
    add_box({ -2.45f, 1.85f, -4.75f }, { 0.24f, 3.2f, 0.24f }, cyan_emissive, false, false);
    add_box({ 2.45f, 1.85f, -4.75f }, { 0.24f, 3.2f, 0.24f }, amber_emissive, false, false);

    // ---- THREE HERO SCULPTURES ----

    // 1. Center Hero: Dispersive crystal orb floating inside a brushed anisotropic gold halo
    add_box({ 0.0f, 0.52f, -0.8f }, { 0.86f, 0.76f, 0.86f }, dark_slate, true, true);
    add_box({ 0.0f, 0.915f, -0.8f }, { 0.90f, 0.03f, 0.90f }, brass, false, true);

    MeshStandardMaterial* crystal_mat = drie::mesh_standard_material(
        color: 0xffffff,
        roughness: 0.01f,
        transmission: 1.0f,
        thickness: 0.32f,
        ior: 1.48f,
        attenuation_color: 0xeaf8ff,
        attenuation_distance: 3.5f
    );
    crystal_mat.dispersion = 6.0f;
    crystal_orb = drie::mesh(drie::sphere_geometry(0.50f, 64, 40), crystal_mat);
    crystal_orb.position = { 0.0f, 1.62f, -0.8f };
    crystal_orb.cast_shadow = true;
    scene.add(crystal_orb);

    gold_halo = drie::mesh(drie::torus_geometry(0.76f, 0.045f, 28, 96), brass);
    gold_halo.position = { 0.0f, 1.62f, -0.8f };
    gold_halo.cast_shadow = true;
    gold_halo.receive_shadow = true;
    scene.add(gold_halo);

    // 2. Left Hero: Live animated drie::nodes molten-veined obsidian torus
    add_box({ -1.72f, 0.44f, -0.45f }, { 0.74f, 0.60f, 0.74f }, concrete, true, true);
    node_torus = drie::mesh(drie::torus_geometry(0.42f, 0.16f, 36, 96), molten_node_material());
    node_torus.position = { -1.72f, 1.38f, -0.45f };
    node_torus.cast_shadow = true;
    node_torus.receive_shadow = true;
    scene.add(node_torus);

    // 3. Right Hero: Iridescent clearcoat sphere atop a brushed anisotropic champagne column
    MeshStandardMaterial* brushed_cyl = drie::mesh_standard_material(color: 0xd8c8a8, roughness: 0.28f, metalness: 1.0f);
    brushed_cyl.anisotropy = 0.95f;
    Mesh* cyl = drie::mesh(drie::cylinder_geometry(0.34f, 0.38f, 0.82f, 48, 4), brushed_cyl);
    cyl.position = { 1.72f, 0.55f, -0.45f };
    cyl.cast_shadow = true;
    cyl.receive_shadow = true;
    scene.add(cyl);

    MeshStandardMaterial* irid_mat = drie::mesh_standard_material(color: 0x7c8aa4, roughness: 0.16f, metalness: 0.80f);
    irid_mat.iridescence = 1.0f;
    irid_mat.iridescence_ior = 1.45f;
    irid_mat.iridescence_thickness_min = 180;
    irid_mat.iridescence_thickness_max = 720;
    irid_mat.iridescence_thickness_map = iridescence_ramp();
    irid_mat.clearcoat = 1.0f;
    irid_mat.clearcoat_roughness = 0.03f;
    Mesh* irid_ball = drie::mesh(drie::sphere_geometry(0.42f, 64, 36), irid_mat);
    irid_ball.position = { 1.72f, 1.38f, -0.45f };
    irid_ball.cast_shadow = true;
    irid_ball.receive_shadow = true;
    scene.add(irid_ball);

    // ---- DYNAMIC LIGHTS (VSM soft shadows + volumetric fog scatter) ----

    // Sweeping key sunbeam slicing through the concrete colonnade onto the colored walls
    key_spot = drie::spot_light(0xfff1dc, intensity: 54.0f, distance: 18.0f, angle: 0.56f, penumbra: 0.30f, decay: 2.0f);
    key_spot.position = { -1.2f, 4.15f, 2.2f };
    key_spot.target.position = { -3.6f, 1.3f, -1.2f };
    key_spot.cast_shadow = true;
    key_spot.shadow.map_size = 1024;
    key_spot.shadow.near = 0.4f;
    key_spot.shadow.far = 16.0f;
    key_spot.shadow.radius = 2.5f;
    key_spot.fog_glow = 2.4f;
    scene.add(key_spot);
    scene.add(key_spot.target);

    // Cool counter-rim spotlight washing the right cobalt baffle & iridescent sphere
    rim_spot = drie::spot_light(0xcce4ff, intensity: 36.0f, distance: 16.0f, angle: 0.62f, penumbra: 0.35f, decay: 2.0f);
    rim_spot.position = { 0.8f, 4.10f, 2.2f };
    rim_spot.target.position = { 2.8f, 1.3f, -0.6f };
    rim_spot.cast_shadow = true;
    rim_spot.shadow.map_size = 1024;
    rim_spot.shadow.near = 0.4f;
    rim_spot.shadow.far = 14.0f;
    rim_spot.shadow.radius = 2.5f;
    rim_spot.fog_glow = 1.2f;
    scene.add(rim_spot);
    scene.add(rim_spot.target);

    // Hidden rear apse spotlight bouncing warm golden light forward through the archway
    apse_spot = drie::spot_light(0xffe0a8, intensity: 56.0f, distance: 12.0f, angle: 0.78f, penumbra: 0.40f, decay: 2.0f);
    apse_spot.position = { 0.0f, 4.05f, -4.45f };
    apse_spot.target.position = { 0.0f, 1.5f, -5.85f };
    apse_spot.fog_glow = 1.6f;
    scene.add(apse_spot);
    scene.add(apse_spot.target);

    scene.update_matrix_world();

    // ---- RADIANCE CASCADES (both Split sparse & Dense uniform on exact same volume) ----
    Vector3 bmin = { -5.2f, -0.2f, -6.2f };
    Vector3 bmax = { 5.2f, 4.6f, 5.2f };

    split_rc = drie::radiance_cascades_sparse(bmin, bmax, spacing: 0.25f, levels: 4, max_probes: 16384, hash_cap: 65536);
    split_rc.sky_from_scene = false;
    split_rc.sky_color = { 0.004f, 0.005f, 0.009f };
    split_rc.ground_color = { 0.003f, 0.003f, 0.004f };
    split_rc.update_every = 1;
    split_rc.blend = 0.12f;
    split_rc.voxelize(scene);

    dense_rc = drie::radiance_cascades(bmin, bmax, spacing: 0.25f, levels: 4);
    dense_rc.sky_from_scene = false;
    dense_rc.sky_color = { 0.004f, 0.005f, 0.009f };
    dense_rc.ground_color = { 0.003f, 0.003f, 0.004f };
    dense_rc.update_every = 1;
    dense_rc.blend = 0.12f;
    dense_rc.voxelize(scene);

    camera = drie::perspective_camera(56, (float)W / H, 0.08f, 50);
    camera.position = { -1.55f, 1.62f, 4.25f };
    camera.update_matrix_world();
    camera.look_at_vec({ 0.45f, 1.40f, -1.1f });
}

fn void set_mode(RcMode m)
{
    mode = m;
    if (mode == DENSE) dense_rc.reset();
    if (mode == SPLIT) split_rc.reset();
}

fn void update_scene(double dt)
{
    if (sweep) sim_time += dt;
    float t = (float)sim_time;

    // Smoothly sweep the key beam between the vermilion left wall and cobalt right wall
    float sx = math::sin(t * 0.55f - 0.9f);
    key_spot.position = { -1.2f + sx * 1.4f, 4.15f, 2.2f };
    key_spot.target.position = { sx * 4.5f, 1.15f + 0.35f * math::cos(t * 0.4f), -1.2f };
    key_spot.intensity = lights_on ? 54.0f : 0.0f;
    rim_spot.intensity = lights_on ? 36.0f : 0.0f;
    apse_spot.intensity = lights_on ? 56.0f : 0.0f;

    // Gentle sculptural motion
    gold_halo.rotation.x = 0.45f + math::sin(t * 0.7f) * 0.35f;
    gold_halo.rotation.y = t * 0.55f;
    crystal_orb.position.y = 1.62f + math::sin(t * 1.1f) * 0.05f;
    node_torus.rotation.y = t * 0.45f;
    node_torus.rotation.x = 0.35f + math::sin(t * 0.5f) * 0.18f;
    pulse_u.set(0.85f + 0.25f * math::sin(t * 1.8f));

    scene.fog = fog_on ? vol_fog : null;
    floor_mat.ssr = refl_on ? 1 : 0;
    mirror_mat.reflector = refl_on ? wall_mirror : null;

    switch (mode)
    {
        case SPLIT:
            scene.radiance = split_rc;
        case DENSE:
            scene.radiance = dense_rc;
            dense_rc.update(scene);
        case OFF:
            scene.radiance = null;
    }
}

fn void ensure_targets(int w, int h)
{
    if (!scene_rt)
    {
        scene_rt = drie::render_target(w, h, 1, RGBA16F, mrt: true);
        back_rt = drie::render_target(w / 2, h / 2, 1, R16F);
        ao = postprocessing::gtao_pass(camera);
        bloom = postprocessing::unreal_bloom_pass(strength: 0.38f, radius: 0.55f, threshold: 1.0f, levels: 6);
        taa = postprocessing::taa_pass();
        output = postprocessing::output_pass(renderer);
        camera.aspect = (float)w / (float)h;
        camera.update_projection_matrix();
        has_prev_vp = false;
    }
    else if (scene_rt.width != w || scene_rt.height != h)
    {
        scene_rt.set_size(w, h);
        back_rt.set_size(math::max(w / 2, 1), math::max(h / 2, 1));
        camera.aspect = (float)w / (float)h;
        camera.update_projection_matrix();
        taa.reset = true;
        has_prev_vp = false;
    }
}

fn void render_frame(int w, int h, float dt, RenderTarget* final_target, bool wait_readback = false)
{
    ensure_targets(w, h);

    if (!rc_debug::active)
    {
        taa.jitter(camera);
    }
    else
    {
        camera.jitter = {};
        taa.reset = true;
    }

    if (refl_on && has_prev_vp && !rc_debug::active && gtao_on)
    {
        renderer.set_reflection_source(taa.texture(), ao.ao_target.texture, prev_view_proj, linear: true, back: back_rt.texture);
    }
    else if (refl_on && has_prev_vp && !rc_debug::active)
    {
        renderer.set_reflection_source(taa.texture(), scene_rt.depth_texture, prev_view_proj, linear: true, back: back_rt.texture);
    }
    else
    {
        renderer.clear_reflection_source();
    }

    renderer.set_render_target(scene_rt);
    Vector3 dbg_focus = controls ? controls.target : (Vector3){ 0, 1.38f, -1.2f };
    Camera* rc_cam = rc_debug::update(dt, w, h, dbg_focus, 3.85f);

    if (rc_debug::active && rc_cam)
    {
        if (scene.radiance && scene.radiance.sparse && rc_debug::freeze_alloc)
        {
            scene.radiance.freeze_alloc = false;
            renderer.render(scene, camera);
            scene.radiance.freeze_alloc = true;
            renderer.render(scene, rc_cam);
            scene.radiance.freeze_alloc = false;
        }
        else
        {
            renderer.render(scene, rc_cam);
        }
        output.set_source(scene_rt.texture);
        output.render(final_target);
        if (final_target)
        {
            Vector3 look = controls ? (controls.target - camera.position).normalize() : (Vector3){ 0, 0, -1 };
            rc_debug::draw_target(final_target, scene.radiance, MODE_NAMES[mode.ordinal], w, h, camera.position, look);
        }
        drie::commit();
        rc_debug::track(scene.radiance, wait_now: wait_readback);
        return;
    }

    if (refl_on) renderer.render_reflector(wall_mirror, scene, camera);
    renderer.set_render_target(scene_rt);
    renderer.render(scene, camera);
    if (refl_on) renderer.render_back_faces(back_rt);
    if (gtao_on) ao.render(scene_rt);
    if (bloom_on) bloom.render(scene_rt);
    taa.blend = 0.12f;
    taa.render(scene_rt);

    output.set_source(taa.texture());
    output.render(final_target);
    drie::commit();
    rc_debug::track(scene.radiance, wait_now: wait_readback);

    camera.update_matrix_world();
    prev_view_proj = camera.projection_matrix.mul(camera.matrix_world.invert());
    has_prev_vp = true;
}

fn void animate(double time)
{
    frame_count++;
    double dt = last_frame_t < 0 ? 1.0 / 60.0 : math::clamp((time - last_frame_t) * 0.001, 0.0, 0.1);
    last_frame_t = time;

    int win_w = math::max(sapp::width(), 1);
    int win_h = math::max(sapp::height(), 1);
    ulong t0 = stm::now();

    update_scene(dt);
    if (controls)
    {
        controls.enabled = !rc_debug::active;
        if (!rc_debug::active) controls.update();
    }
    render_frame(win_w, win_h, (float)dt, null);

    double ms = stm::ms(stm::since(t0));
    smooth_ms = smooth_ms * 0.9 + ms * 0.1;
}

fn void on_event(Event* e)
{
    if (e.type == KEY_DOWN && !e.key_repeat)
    {
        if (e.key_code == TAB)
        {
            if (last_tab_frame != frame_count)
            {
                last_tab_frame = frame_count;
                set_mode(mode == SPLIT ? DENSE : (mode == DENSE ? OFF : SPLIT));
            }
            return;
        }
        if (e.key_code == L)
        {
            lights_on = !lights_on;
            return;
        }
    }
    Vector3 dbg_focus = controls ? controls.target : (Vector3){ 0, 1.38f, -1.2f };
    if (rc_debug::handle_event(e, dbg_focus, 3.85f)) return;
    if (e.type != KEY_DOWN || e.key_repeat) return;
    switch (e.key_code)
    {
        case NUM_1: set_mode(SPLIT);
        case NUM_2: set_mode(DENSE);
        case NUM_3: set_mode(OFF);
        case V:     fog_on = !fog_on;
        case R:     refl_on = !refl_on;
        case G:     gtao_on = !gtao_on;
        case B:     bloom_on = !bloom_on;
        case SPACE: sweep = !sweep;
        default: {}
    }
}

fn void hud()
{
    if (rc_debug::active)
    {
        Vector3 look = controls ? (controls.target - camera.position).normalize() : (Vector3){ 0, 0, -1 };
        int win_w = math::max(sapp::width(), 1);
        int win_h = math::max(sapp::height(), 1);
        int ui_h = H;
        int ui_w = math::max((int)math::round((float)ui_h * (float)win_w / (float)win_h), 640);
        rc_debug::draw_window(scene.radiance, MODE_NAMES[mode.ordinal], ui_w, ui_h, camera.position, look);
        return;
    }
    ui::setup();
    float scale = math::max(1.0f, math::round(sapp::heightf() / 640));
    sdtx::set_context(sdtx::default_context());
    sdtx::canvas(sapp::widthf() / scale, sapp::heightf() / scale);
    sdtx::origin(1.5f, 1.5f);
    sdtx::font(ui::FONT_ORIC);
    @pool() {
        sdtx::color3f(0.35f, 0.92f, 1.0f);
        sdtx::puts(string::tformat("THE LIGHT SANCTUARY  [GI: %s]   %.1f ms (%.0f fps)\n\n",
            MODE_NAMES[mode.ordinal], smooth_ms, 1000.0 / math::max(smooth_ms, 0.1)).zstr_tcopy());
        sdtx::color3f(0.90f, 0.90f, 0.92f);
        sdtx::puts(string::tformat("1-3/Tab: GI   V: Fog[%s]  R: SSR+Mirror[%s]  G: GTAO[%s]  B: Bloom[%s]\n",
            fog_on ? "ON" : "OFF", refl_on ? "ON" : "OFF", gtao_on ? "ON" : "OFF", bloom_on ? "ON" : "OFF").zstr_tcopy());
        sdtx::puts(string::tformat("L: Spotlights[%s]   Space: Sweep[%s]   F3: RC Probe Debugger\n",
            lights_on ? "ON" : "EMISSIVES", sweep ? "ON" : "PAUSED").zstr_tcopy());
        sdtx::color3f(0.65f, 0.70f, 0.76f);
        sdtx::puts("Drag: orbit   Scroll: zoom   Right-drag: pan");
    };
    sdtx::draw();
}

fn int main(String[] args)
{
    String png_path;
    bool hdr = false;
    bool p3 = false;
    for (usz i = 1; i < args.len; i++)
    {
        if (args[i] == "--png" && i + 1 < args.len)
        {
            png_path = args[++i];
        }
        else if (args[i] == "--mode" && i + 1 < args.len)
        {
            String m = args[++i];
            if (m == "dense") { mode = DENSE; }
            else if (m == "off") { mode = OFF; }
            else { mode = SPLIT; }
        }
        else if (args[i] == "--no-lights")
        {
            lights_on = false;
        }
        else if (args[i] == "--hdr")
        {
            hdr = true;
        }
        else if (args[i] == "--p3")
        {
            p3 = true;
        }
        else if (args[i] == "--rc-debug-3d")
        {
            rc_debug::active = true;
            rc_debug::set_3d(true);
        }
        else
        {
            io::eprintfn("unknown option %s", args[i]);
        }
    }

    if (!png_path)
    {
        renderer = drie::renderer(antialias: false, width: W, height: H, title: "drie: the light sanctuary");
        renderer.shadow_type = VSM;
        renderer.tone_mapping = ACES_FILMIC_TONE_MAPPING;
        renderer.tone_mapping_exposure = 1.15f;
        if (p3 || hdr) renderer.output_color_space = DISPLAY_P3_COLOR_SPACE;
        if (hdr)
        {
            renderer.hdr_window = true;
            renderer.hdr_output = true;
        }
        build_scene();
        controls = controls::orbit_controls(camera);
        controls.target = { 0.45f, 1.40f, -1.1f };
        drie::add_event_listener(&on_event);
        drie::add_overlay(&hud);
        renderer.set_animation_loop(&animate);
        return 0;
    }

    renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.shadow_type = VSM;
    renderer.tone_mapping = ACES_FILMIC_TONE_MAPPING;
    renderer.tone_mapping_exposure = 1.15f;
    build_scene();
    RenderTarget* toned = drie::render_target(W, H, 1, RGBA8, false);
    for (int f = 0; f < 32; f++)
    {
        update_scene(f == 0 ? 0.0 : 0.0);
        render_frame(W, H, 0.0f, toned, wait_readback: f == 31);
    }
    char[] px = mem::new_array(char, (usz)W * H * 4);
    readback::read_image(toned.texture_image, W, H, px.ptr);
    if (!png::write(png_path, px, W, H))
    {
        io::eprintfn("could not write %s", png_path);
        return 1;
    }
    io::printfn("saved %s (%s)", png_path, MODE_NAMES[mode.ordinal]);
    return 0;
}
