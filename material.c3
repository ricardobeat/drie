// Materials. Constructors take named arguments, mirroring three's parameter objects:
//   three::mesh_standard_material(color: 0x44aa88, roughness: 0.4)
module three;

enum Side
{
    FRONT_SIDE,
    BACK_SIDE,
    DOUBLE_SIDE,
}

// How a see-through material's pixels are put over what is behind them (like three's
// `blending`): mixed in by their opacity, or added to it, as light is.
enum Blending
{
    NORMAL_BLENDING,
    ADDITIVE_BLENDING,
}

// Values match the shader's `mode`.
constdef MaterialType : int
{
    BASIC = 0,
    LAMBERT = 1,
    PHONG = 2,
    STANDARD = 3,
    NORMAL = 4,
}

struct Material
{
    MaterialType type;
    Color color;
    float opacity;
    bool transparent;
    Blending blending;
    Side side;
    bool wireframe;
    bool visible;
    bool fog;
    bool depth_test;
    bool depth_write;
    Color emissive;
    float emissive_intensity;
    // Phong
    Color specular;
    float shininess;
    // Standard (PBR)
    float roughness;
    float metalness;
    // multiply color by the geometry's vertex colors
    bool vertex_colors;
    // Color texture, multiplied with `color`. Its alpha channel is the opacity.
    Texture* map;
    // Tangent-space normal map (OpenGL convention, +Y up) and its strength.
    Texture* normal_map;
    float normal_scale;
    // Packed ambient occlusion (R), roughness (G) and metalness (B), as in glTF's ORM textures.
    // Roughness and metalness are multiplied with the material's values.
    Texture* arm_map;
    // Discard pixels with alpha below this (0 = off). Cutout edges are antialiased with MSAA.
    float alpha_test;
    // Extensions:
    // Light thin double-sided surfaces (grass, leaves) from both sides, with light wrapping around.
    bool foliage;
    // Transparent, but reflections and highlights stay fully visible. Use with `transparent`.
    bool glass;
    // Reflections of the surroundings: an equirectangular map, for example from
    // Renderer.capture_environment (like three's envMap). Rougher surfaces blur it.
    Texture* env_map;
    float env_intensity;
    // Box projection for env_map (like Unity's reflection probes): the map was captured at
    // env_position inside the box env_box_min..max, usually a room. Reflections of its walls
    // then stay in place as the viewer moves. Off when env_box_min == env_box_max.
    Vector3 env_position;
    Vector3 env_box_min;
    Vector3 env_box_max;
    // Screen-space reflections (see Renderer.set_reflection_source): the surface mirrors what
    // was on screen around it, more at a glancing angle. 0 = off, 1 = as a varnished surface
    // would; roughness blurs it.
    float ssr;
    // Dirt, stains, runs of water and flaking paint, as on the walls, floor and ceiling of a
    // neglected room: 0 = clean, 1 = filthy. It is worked out from where the surface is, so
    // it needs to know the height of the room's floor (grime_floor) and how high the room is
    // (grime_height, 2.6 m when 0); grime_seed moves the pattern.
    float grime;
    float grime_floor;
    float grime_height;
    float grime_seed;
    // Glass only: how dark the viewer's own silhouette shows in reflections (0 = not at all).
    float reflect_viewer;
    // Vertex colors are light baked into the geometry (from lamps that don't need to be real
    // lights) instead of a tint: they light the surface with this strength. 0 = off.
    float vertex_light;
    // Draws the surface as a window onto this render target, pixel for pixel at the same screen
    // position, unlit (for portals and mirrors rendered from another camera).
    RenderTarget* screen_target;
    // Wind: sideways movement in meters per meter of height above the model's origin, and its speed.
    float sway;
    float sway_speed;
    // How far the material's vertices give way to the renderer's pusher (0 = not at all, 1 =
    // as a branch would). The part of a plant near its stem stays where it is.
    float push;
}

struct MeshBasicMaterial { inline Material material; }
struct MeshLambertMaterial { inline Material material; }
struct MeshPhongMaterial { inline Material material; }
struct MeshStandardMaterial { inline Material material; }
struct MeshNormalMaterial { inline Material material; }

fn void Material.init_base(&self, MaterialType type, Color color, float opacity, bool transparent, Side side,
                           bool wireframe, bool fog)
{
    self.type = type;
    self.color = color;
    self.opacity = opacity;
    self.transparent = transparent;
    self.side = side;
    self.wireframe = wireframe;
    self.fog = fog;
    self.visible = true;
    self.depth_test = true;
    self.depth_write = true;
    self.emissive_intensity = 1;
    self.specular = 0x111111;
    self.shininess = 30;
    self.roughness = 1;
    self.normal_scale = 1;
    self.sway_speed = 1;
}

// Unlit, flat color.
fn MeshBasicMaterial* mesh_basic_material(Color color = 0xffffff, float opacity = 1, bool transparent = false,
                                          Side side = FRONT_SIDE, bool wireframe = false, bool fog = true, bool vertex_colors = false,
        Texture* map = null)
{
    MeshBasicMaterial* m = mem::new(MeshBasicMaterial);
    m.init_base(BASIC, color, opacity, transparent, side, wireframe, fog);
    m.vertex_colors = vertex_colors;
    m.map = map;
    return m;
}

// Diffuse-only lighting.
fn MeshLambertMaterial* mesh_lambert_material(Color color = 0xffffff, Color emissive = 0x000000,
                                              float emissive_intensity = 1, float opacity = 1, bool transparent = false,
                                              Side side = FRONT_SIDE, bool wireframe = false, bool fog = true, bool vertex_colors = false,
        Texture* map = null)
{
    MeshLambertMaterial* m = mem::new(MeshLambertMaterial);
    m.init_base(LAMBERT, color, opacity, transparent, side, wireframe, fog);
    m.vertex_colors = vertex_colors;
    m.map = map;
    m.emissive = emissive;
    m.emissive_intensity = emissive_intensity;
    return m;
}

// Blinn-Phong with specular highlights.
fn MeshPhongMaterial* mesh_phong_material(Color color = 0xffffff, Color specular = 0x111111, float shininess = 30,
                                          Color emissive = 0x000000, float emissive_intensity = 1, float opacity = 1,
                                          bool transparent = false, Side side = FRONT_SIDE, bool wireframe = false,
                                          bool fog = true, bool vertex_colors = false,
        Texture* map = null)
{
    MeshPhongMaterial* m = mem::new(MeshPhongMaterial);
    m.init_base(PHONG, color, opacity, transparent, side, wireframe, fog);
    m.vertex_colors = vertex_colors;
    m.map = map;
    m.specular = specular;
    m.shininess = shininess;
    m.emissive = emissive;
    m.emissive_intensity = emissive_intensity;
    return m;
}

// Physically based (metalness / roughness).
fn MeshStandardMaterial* mesh_standard_material(Color color = 0xffffff, float roughness = 1, float metalness = 0,
                                                Color emissive = 0x000000, float emissive_intensity = 1,
                                                float opacity = 1, bool transparent = false, Side side = FRONT_SIDE,
                                                bool wireframe = false, bool fog = true, bool vertex_colors = false,
        Texture* map = null)
{
    MeshStandardMaterial* m = mem::new(MeshStandardMaterial);
    m.init_base(STANDARD, color, opacity, transparent, side, wireframe, fog);
    m.vertex_colors = vertex_colors;
    m.map = map;
    m.roughness = roughness;
    m.metalness = metalness;
    m.emissive = emissive;
    m.emissive_intensity = emissive_intensity;
    return m;
}

// Colors surfaces by their normal direction.
fn MeshNormalMaterial* mesh_normal_material(float opacity = 1, bool transparent = false, Side side = FRONT_SIDE,
                                            bool wireframe = false)
{
    MeshNormalMaterial* m = mem::new(MeshNormalMaterial);
    m.init_base(NORMAL, 0xffffff, opacity, transparent, side, wireframe, false);
    return m;
}
