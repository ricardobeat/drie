// Materials. Constructors take named arguments, mirroring three's parameter objects:
//   drie::mesh_standard_material(color: 0x44aa88, roughness: 0.4)
module drie;

enum Side {
    FRONT_SIDE,
    BACK_SIDE,
    DOUBLE_SIDE,
}

// How a material's pixels are put over what is behind them (three's `blending`): mixed in by
// their opacity (for a `transparent` material; an opaque one covers what is behind), added
// to it as light is, taken from it, multiplying it, as the material's blend_* say
// (CUSTOM_BLENDING), or not at all, covering it. With premultiplied_alpha the material's
// colors come out multiplied by their opacity, and the blending takes them so (three's).
enum Blending {
    NORMAL_BLENDING,
    ADDITIVE_BLENDING,
    SUBTRACTIVE_BLENDING,
    MULTIPLY_BLENDING,
    CUSTOM_BLENDING,
    NO_BLENDING,
}

// CUSTOM_BLENDING's: how the material's color (the source) and what is behind it (the
// destination) are weighed (three's blendSrc, blendDst and their factors) and combined
// (blendEquation).
enum BlendEquation {
    ADD_EQUATION,
    SUBTRACT_EQUATION,
    REVERSE_SUBTRACT_EQUATION,
    MIN_EQUATION,
    MAX_EQUATION,
}

enum BlendFactor {
    ZERO_FACTOR,
    ONE_FACTOR,
    SRC_COLOR_FACTOR,
    ONE_MINUS_SRC_COLOR_FACTOR,
    SRC_ALPHA_FACTOR,
    ONE_MINUS_SRC_ALPHA_FACTOR,
    DST_ALPHA_FACTOR,
    ONE_MINUS_DST_ALPHA_FACTOR,
    DST_COLOR_FACTOR,
    ONE_MINUS_DST_COLOR_FACTOR,
    SRC_ALPHA_SATURATE_FACTOR,
}

// How a transmission material's volume bends the rays through it (Material.refraction_model;
// not in three.js, as Unity HDRP's refraction models). SPHERE: every ray crosses `thickness`
// of it, whatever its angle (three's). PLANE: it is a wall `thickness` thick, a pane or a
// glass's side: a ray that meets it at a slant crosses more of it (thickness / cos), so it
// is shifted further and tinted more toward the edges of a curved wall, as real glass is.
enum RefractionModel {
    SPHERE,
    PLANE,
}

// How a basic, lambert or phong material's env_map is put over its light (three's `combine`):
// the light times the reflection, the reflection in place of the light, or the two added,
// each by Material.reflectivity.
enum Combine {
    MULTIPLY_OPERATION,
    MIX_OPERATION,
    ADD_OPERATION,
}

// Values match the shader's `mode`.
constdef MaterialType : int {
    BASIC    = 0,
    LAMBERT  = 1,
    PHONG    = 2,
    STANDARD = 3,
    NORMAL   = 4,
    MATCAP   = 5,
    TOON     = 6,
    DEPTH    = 7,
}

// How MeshDepthMaterial writes the depth (three's depthPacking): BASIC_DEPTH_PACKING as a
// gray (white near), or spread over the channels for more than 8 bits.
enum DepthPacking {
    BASIC_DEPTH_PACKING,
    RGBA_DEPTH_PACKING,
    RGB_DEPTH_PACKING,
    RG_DEPTH_PACKING,
}

struct Material {
    MaterialType type;
    Color color;
    float opacity;
    bool transparent;
    Blending blending;
    bool premultiplied_alpha;
    // CUSTOM_BLENDING: color = src * blend_src (equation) dst * blend_dst, and alpha the same
    // with the _alpha ones (three's blendEquation, blendSrc, blendDst, blendEquationAlpha,
    // blendSrcAlpha and blendDstAlpha)
    BlendEquation blend_equation;
    BlendFactor blend_src;
    BlendFactor blend_dst;
    BlendEquation blend_equation_alpha;
    BlendFactor blend_src_alpha;
    BlendFactor blend_dst_alpha;
    Side side;
    bool wireframe;
    bool visible;
    bool fog;
    bool depth_test;
    bool depth_write;
    // Its colors go through the renderer's tone mapping (three's toneMapped).
    bool tone_mapped;
    Color emissive;
    float emissive_intensity;
    // The color map glows too: emissive * emissive_intensity is multiplied by it (three's
    // emissiveMap, when it is the same image as the map).
    bool map_emissive;
    // Multiplies `color`, for colors brighter than white (a three Color can hold any number;
    // drie's has 8 bits a channel). 0 counts as 1.
    float color_scale;
    // The lighting channels whose lights light it (bits; 0: every light): see Light.channels.
    // Three's lightsNode (lights([...])) as a channel of its own.
    uint light_channels;
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
    // Tangent-space normal map (OpenGL convention, +Y up) and its strength, x and y (three's
    // normalScale). The tangent frame is the geometry's tangents, or without them (three's
    // derivative tangents) one where +Y is the way the uv's v grows: a glTF model's normals,
    // made with its v growing down the image, then need y negated (as the glTF loader does).
    Texture* normal_map;
    Vector2 normal_scale;
    // Packed ambient occlusion (R), roughness (G) and metalness (B), as in glTF's ORM textures.
    // Roughness and metalness are multiplied with the material's values.
    Texture* arm_map;
    // The same, a map at a time, as three has them (without an arm_map): roughness from the
    // green channel of roughness_map, metalness from the blue of metalness_map (glTF's
    // metallicRoughnessTexture is both: when they are different images, metalness_map is not
    // used), and ambient occlusion from the red of ao_map, ao_intensity strong (three's
    // aoMapIntensity; 0: none).
    Texture* roughness_map;
    Texture* metalness_map;
    Texture* ao_map;
    float ao_intensity;
    // Parallax occlusion mapping (not in three.js; three's displacementMap moves the vertices,
    // this only the texture): a height map (r: 1 = the top of the surface, 0 = parallax_scale
    // meters under it), in the same uv as the other maps. The view ray is marched into it and
    // the color, normal and arm maps are all read where it meets it, so stones and ruts stand
    // up and hide each other. It fades out toward parallax_distance meters away (0: 30 m), and
    // with parallax_shadow (0..1) the bumps shade each other from the first directional light.
    Texture* parallax_map;
    float parallax_scale;
    float parallax_distance;
    float parallax_shadow;
    // Hex tiling (not in three.js; Mikkelsen's, 2022, as game engines have it): the maps are
    // read in hexagonal patches, each from its own random place in the texture and blended into
    // the next where they meet, so that a texture repeated over a large surface doesn't show its
    // repeats. hex_tiling is how wide the patches are, in the color map's uv (0: off; 0.5: half
    // the texture across). They are moved, not turned: what runs down the texture (stains,
    // drips) still runs down. Three reads of each map instead of one. Not with a parallax_map or
    // alpha_test, nor for the maps read from the second uv set (a light map).
    float hex_tiling;
    // A height map (red, 1 = highest) that tilts the normal by how it changes from pixel to
    // pixel, bump_scale strong (three's bumpMap and bumpScale), after the normal map.
    Texture* bump_map;
    float bump_scale;
    // A height map (red) that moves the vertices out along their normals by its red times
    // displacement_scale, plus displacement_bias (three's displacementMap, displacementScale
    // and displacementBias). The normals stay as they are, and so do the bounds the frustum
    // culling goes by. The map is read at each vertex on the CPU, once for each geometry and
    // placement of the map, so it keeps its pixels (Texture.keep_pixels, which the renderer
    // sets when it first draws the material: before the upload, unless another material has
    // drawn with the texture already). An 8-bit image only.
    Texture* displacement_map;
    float displacement_scale;
    float displacement_bias;
    // MeshMatcapMaterial's image of a lit sphere, read where the normal points as the viewer
    // sees it (three's matcap; without one, a gray gradient from top to bottom).
    Texture* matcap;
    // MeshToonMaterial's steps of light: the red of this map, read across by how squarely the
    // light falls (three's gradientMap; without one, two steps). Read texel by texel when the
    // texture's `nearest` is on, as three's NearestFilter.
    Texture* gradient_map;
    // MeshDepthMaterial's packing of the depth.
    DepthPacking depth_packing;
    // Opacity from the green channel of this map, times `opacity` (three's alphaMap).
    Texture* alpha_map;
    // What glows: emissive * emissive_intensity times this map's rgb (sRGB; three's emissiveMap).
    Texture* emissive_map;
    // Baked light (three's lightMap), linear rgb, light_map_intensity strong: added to the
    // indirect light of lit materials; for BASIC the indirect light (1) becomes map / PI.
    // Read from the uv set of its `channel` (1: the second, uv1).
    Texture* light_map;
    float light_map_intensity;
    // Discard pixels with alpha below this (0 = off). Cutout edges are antialiased with MSAA.
    float alpha_test;
    // Extensions:
    // Light thin double-sided surfaces (grass, leaves) from both sides, with light wrapping around.
    bool foliage;
    // Transparent, but reflections and highlights stay fully visible. Use with `transparent`.
    bool glass;
    // Light passing through: how much of what stands behind the surface shows through it, bent
    // as it crosses (KHR_materials_transmission). The renderer draws the scene behind into a
    // target of its own first, so it costs a second pass over the opaque scene. 0 = off.
    // Rough surfaces blur what they show (roughness, as three: the picture's mip levels).
    float transmission;
    // times the red channel of this map
    Texture* transmission_map;
    // How far (m) a ray bends as it travels through it: light leaves the surface shifted by the
    // part of the material it crosses. Thin glass wants a few millimeters. In the model's own
    // meters: the object's scale scales it.
    float thickness;
    // times the green channel of this map
    Texture* thickness_map;
    // SPHERE (three's) or PLANE: see RefractionModel.
    RefractionModel refraction_model;
    // Index of refraction: how strongly it bends light (1.5: glass, 1.33: water). It also sets
    // how much a dielectric reflects head on: ((ior - 1) / (ior + 1))², 4% for 1.5.
    float ior;
    // The strength and color of a standard material's reflections where it isn't metal
    // (KHR_materials_specular, three's specularIntensity and specularColor): the reflectance
    // head on (from `ior`) is tinted by specular_color, linear and allowed above 1, times the
    // rgb of specular_color_map (sRGB), and scaled by specular_intensity times the alpha of
    // specular_intensity_map. 1 and white: as without them.
    float specular_intensity;
    Vector3 specular_color;
    Texture* specular_color_map;
    Texture* specular_intensity_map;
    // A clear coat over a standard material (KHR_materials_clearcoat, three's clearcoat): a
    // varnish with highlights and reflections of its own, clearcoat_roughness rough (0.045 at
    // the least), that lets through what it does not reflect. Its normal is the surface's own,
    // without the normal_map, or from clearcoat_normal_map (scaled by clearcoat_normal_scale,
    // x and y, as three's clearcoatNormalScale). 0 = none; times the red channel of
    // clearcoat_map, and its roughness times the green of clearcoat_roughness_map.
    float clearcoat;
    float clearcoat_roughness;
    Texture* clearcoat_map;
    Texture* clearcoat_roughness_map;
    Texture* clearcoat_normal_map;
    Vector2 clearcoat_normal_scale;
    // Sheen (KHR_materials_sheen, three's sheen): the soft glow at the edges of cloth, from
    // fibers on the surface. sheen_color (linear, times the rgb of sheen_color_map, sRGB) times
    // sheen is how much they reflect, sheen_roughness (times the alpha of sheen_roughness_map)
    // how widely. 0 sheen or a black sheen_color = none.
    float sheen;
    Vector3 sheen_color;
    float sheen_roughness;
    Texture* sheen_color_map;
    Texture* sheen_roughness_map;
    // Iridescence (KHR_materials_iridescence, three's iridescence): a thin film on the surface,
    // like soap or oil on water, whose reflections shift through rainbow colors with the angle.
    // The film's IOR is iridescence_ior and it is iridescence_thickness_max nanometers thick,
    // or with iridescence_thickness_map from _min (green 0) to _max (green 1). 0 = none; times
    // the red channel of iridescence_map.
    float iridescence;
    float iridescence_ior;
    float iridescence_thickness_min;
    float iridescence_thickness_max;
    Texture* iridescence_map;
    Texture* iridescence_thickness_map;
    // Anisotropy (KHR_materials_anisotropy, three's anisotropy): highlights and reflections
    // drawn out along a direction of the surface, as on brushed metal. 0 = none, 1 = the
    // most; the direction is the tangent's (the way the uv's u grows; see normal_map), turned
    // anisotropy_rotation radians toward the bitangent, and anisotropy_map's rg (a direction
    // around 0.5, 0.5) turns it further and its b scales the strength. Needs roughness.
    float anisotropy;
    float anisotropy_rotation;
    Texture* anisotropy_map;
    // How much the refraction differs across the red..blue channels (KHR_materials_dispersion):
    // each is bent with an IOR of `ior` + (ior - 1) * 0.025 * dispersion, so edges come out
    // fringed with rainbow colors. 0 = all wavelengths bent alike.
    float dispersion;
    // Light is tinted as it crosses the volume, toward this color: Beer's law over
    // attenuation_distance meters. 0 distance: no tint.
    Color attenuation_color;
    float attenuation_distance;
    // Reflections of the surroundings: an equirectangular map, for example from
    // Renderer.capture_environment (like three's envMap). Rougher surfaces blur it.
    Texture* env_map;
    // On a basic, lambert or phong material, the env_map is three's: what the surface mirrors
    // (or, with env_refraction, what is seen through it, bent by refraction_ratio: the ratio of
    // the two IORs, three's refractionRatio), sharp, put over its light as `combine` says by
    // `reflectivity` (0: not at all) times the red channel of specular_map, which also scales
    // a phong material's highlights (three's specularMap).
    Combine combine;
    float reflectivity;
    bool env_refraction;
    float refraction_ratio;
    Texture* specular_map;
    // Without an env_map: mirrors the scene's reflection probes (Scene.reflection_probes),
    // each point the probe of the box it is in.
    bool env_probes;
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
    // how far (m) its reflected rays go looking for something: 0 = the renderer's reach
    // (set_reflection_source). Short, only what stands on the surface is mirrored, and the rest
    // is left to the environment map or the reflector
    float ssr_reach;
    // A Reflector whose plane the surface lies on: it mirrors that picture of the scene (see
    // reflector.c3), and screen-space reflections (if on) only where they find something
    Reflector* reflector;
    // Vertex colors as occlusion (not in three.js): besides tinting the surface (vertex_colors),
    // their brightness dims all the light it gets, its reflections and highlights included, so
    // that a surface fading into the dark (down a shaft, a pit) goes to true black there
    // whatever lights it or mirrors in it.
    bool vertex_occlusion;
    // Vertex colors are light baked into the geometry (from lamps that don't need to be real
    // lights) instead of a tint: they light the surface with this strength. 0 = off.
    float vertex_light;
    // Pushes the vertices out along their normals by this much of the view (clip space per unit
    // of w), the same on screen at any distance: a back-side material with it, drawn over a mesh's
    // own, outlines it (three's OutlineEffect, whose thickness this is). 0: off.
    float outline;
    // Draws the surface as a window onto this render target, pixel for pixel at the same screen
    // position, unlit (for portals and mirrors rendered from another camera).
    RenderTarget* screen_target;
    // Wind: sideways movement in meters per meter of height above the model's origin, and its speed.
    float sway;
    float sway_speed;
    // Dashed lines (line_dashed_material): dashes of dash_size with gaps of gap_size, in units
    // of the distances along the line (Line.compute_line_distances) times dash_scale. Solid
    // when either size is 0.
    float dash_size;
    float gap_size;
    float dash_scale;
    // For the application's own shader hooks (Renderer.shaders, see shader.glsl), which read
    // them as `material_custom[0]` and `[1]`. The engine's own shader ignores them.
    Vec4[2] custom;
    // And a texture for them: a hook reads it with custom_map_texel(uv, dx, dy), at a uv of its
    // own (world meters, say) and its change from one pixel to the next. It takes the
    // material's sampler, its first map's (map, normal_map, arm_map, then the others), and so
    // that one's wrapping.
    Texture* custom_map;
    // Blurred by how far it is out of focus, by its own distance (not three's): its maps are
    // read over the blur of the lens's focus at its depth (Renderer.focus_map, from a
    // BokehPass). For a transparent surface near the lens, a dirty pane, which a depth of field
    // pass sees through: there, the blur goes by what is behind it.
    bool defocus;
    // Node graphs (nodes.c3, drie::nodes; three's NodeMaterial and TSL), each in place of a part
    // of the material where it is set:
    //   color_node     the color, rgb or rgba (instead of color, the color map and vertex colors)
    //   opacity_node   its alpha
    //   normal_node    the normal it is shaded with, in the world (after the normal map)
    //   roughness_node, metalness_node
    //   emissive_node  the glow (instead of emissive and emissive_map)
    //   position_node  the vertex's position in the model (with an instance's matrix), in
    //                  every pass: the depth prepass and the shadow maps see it too
    //   output_node    what comes out, rgba, from what the light made of it
    //                  (nodes::output(); before the fog)
    //   mask_node      a bool: false cuts the fragment out, in every pass (the shadow maps
    //                  too), as three's maskNode
    //   env_node       what the environment reflects, rgb, in nodes::env_direction() at
    //                  nodes::env_roughness() (three's envNode; in place of the environment
    //                  map's reflection, whose light on surfaces stays)
    //   specular_node  a phong material's specular color, rgb (instead of specular)
    //   shininess_node and its shininess
    Node color_node;
    Node opacity_node;
    Node normal_node;
    Node roughness_node;
    Node metalness_node;
    Node emissive_node;
    Node position_node;
    Node output_node;
    Node mask_node;
    Node env_node;
    Node specular_node;
    Node shininess_node;
    // The shaders built for them (at the first draw, and again when one changes).
    NodeBuild* node_build;
}

// A copy of the material to change on its own (like three's Material.clone). Textures are
// shared.
fn Material* Material.clone(&self) {
    Material* m = mem::new(Material);
    *m = *self;
    // (its node shaders: its own build, sharing the program)
    m.node_build = null;
    return m;
}

struct MeshBasicMaterial {
    inline Material material;
}
struct MeshLambertMaterial {
    inline Material material;
}
struct MeshPhongMaterial {
    inline Material material;
}
struct MeshStandardMaterial {
    inline Material material;
}
struct MeshNormalMaterial {
    inline Material material;
}
struct MeshMatcapMaterial {
    inline Material material;
}
struct MeshToonMaterial {
    inline Material material;
}
struct MeshDepthMaterial {
    inline Material material;
}
struct LineBasicMaterial {
    inline Material material;
}
struct LineDashedMaterial {
    inline Material material;
}

fn void Material.init_base(
    &self,
    MaterialType type,
    Color color,
    float opacity,
    bool transparent,
    Side side,
    bool wireframe,
    bool fog
) {
    self.type = type;
    self.color = color;
    self.opacity = opacity;
    self.transparent = transparent;
    self.side = side;
    self.wireframe = wireframe;
    self.fog = fog;
    self.visible = true;
    self.depth_test = true;
    self.tone_mapped = true;
    self.depth_write = true;
    self.emissive_intensity = 1;
    self.specular = 0x111111;
    self.shininess = 30;
    self.roughness = 1;
    self.normal_scale = { 1, 1 };
    self.sway_speed = 1;
    self.ior = 1.5f;
    self.attenuation_color = 0xffffff;
    self.specular_intensity = 1;
    self.specular_color = { 1, 1, 1 };
    self.ao_intensity = 1;
    self.light_map_intensity = 1;
    self.reflectivity = 1;
    self.blend_src = SRC_ALPHA_FACTOR;
    self.blend_dst = ONE_MINUS_SRC_ALPHA_FACTOR;
    self.blend_src_alpha = SRC_ALPHA_FACTOR;
    self.blend_dst_alpha = ONE_MINUS_SRC_ALPHA_FACTOR;
    self.bump_scale = 1;
    self.displacement_scale = 1;
    self.refraction_ratio = 0.98f;
    self.clearcoat_normal_scale = { 1, 1 };
    self.sheen_roughness = 1;
    self.iridescence_ior = 1.3f;
    self.iridescence_thickness_min = 100;
    self.iridescence_thickness_max = 400;
}

// Unlit, flat color.
fn MeshBasicMaterial* mesh_basic_material(
    Color color = 0xffffff,
    float opacity = 1,
    bool transparent = false,
    Side side = FRONT_SIDE,
    bool wireframe = false,
    bool fog = true,
    bool vertex_colors = false,
    Texture* map = null
) {
    MeshBasicMaterial* m = mem::new(MeshBasicMaterial);
    m.init_base(BASIC, color, opacity, transparent, side, wireframe, fog);
    m.vertex_colors = vertex_colors;
    m.map = map;
    return m;
}

// For lines (Line, LineSegments, LineLoop): unlit, flat color, like three's LineBasicMaterial.
// Its linewidth has no counterpart: lines are a pixel wide, as in WebGL.
fn LineBasicMaterial* line_basic_material(
    Color color = 0xffffff,
    float opacity = 1,
    bool transparent = false,
    bool fog = true,
    bool vertex_colors = false,
    Texture* map = null
) {
    LineBasicMaterial* m = mem::new(LineBasicMaterial);
    m.init_base(BASIC, color, opacity, transparent, FRONT_SIDE, false, fog);
    m.vertex_colors = vertex_colors;
    m.map = map;
    return m;
}

// A dashed line_basic_material, like three's LineDashedMaterial (defaults as three's). The
// line needs its distances: call compute_line_distances() on it.
fn LineDashedMaterial* line_dashed_material(
    Color color = 0xffffff,
    float dash_size = 3,
    float gap_size = 1,
    float scale = 1,
    float opacity = 1,
    bool transparent = false,
    bool fog = true,
    bool vertex_colors = false
) {
    LineDashedMaterial* m = mem::new(LineDashedMaterial);
    m.init_base(BASIC, color, opacity, transparent, FRONT_SIDE, false, fog);
    m.vertex_colors = vertex_colors;
    m.dash_size = dash_size;
    m.gap_size = gap_size;
    m.dash_scale = scale;
    return m;
}

// Diffuse-only lighting.
fn MeshLambertMaterial* mesh_lambert_material(
    Color color = 0xffffff,
    Color emissive = 0x000000,
    float emissive_intensity = 1,
    float opacity = 1,
    bool transparent = false,
    Side side = FRONT_SIDE,
    bool wireframe = false,
    bool fog = true,
    bool vertex_colors = false,
    Texture* map = null
) {
    MeshLambertMaterial* m = mem::new(MeshLambertMaterial);
    m.init_base(LAMBERT, color, opacity, transparent, side, wireframe, fog);
    m.vertex_colors = vertex_colors;
    m.map = map;
    m.emissive = emissive;
    m.emissive_intensity = emissive_intensity;
    return m;
}

// Blinn-Phong with specular highlights.
fn MeshPhongMaterial* mesh_phong_material(
    Color color = 0xffffff,
    Color specular = 0x111111,
    float shininess = 30,
    Color emissive = 0x000000,
    float emissive_intensity = 1,
    float opacity = 1,
    bool transparent = false,
    Side side = FRONT_SIDE,
    bool wireframe = false,
    bool fog = true,
    bool vertex_colors = false,
    Texture* map = null
) {
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
fn MeshStandardMaterial* mesh_standard_material(
    Color color = 0xffffff,
    float roughness = 1,
    float metalness = 0,
    Color emissive = 0x000000,
    float emissive_intensity = 1,
    float opacity = 1,
    bool transparent = false,
    Side side = FRONT_SIDE,
    bool wireframe = false,
    bool fog = true,
    bool vertex_colors = false,
    Texture* map = null,
    float transmission = 0,
    float thickness = 0,
    float ior = 1.5f,
    float dispersion = 0,
    Color attenuation_color = 0xffffff,
    float attenuation_distance = 0
) {
    MeshStandardMaterial* m = mem::new(MeshStandardMaterial);
    m.init_base(STANDARD, color, opacity, transparent, side, wireframe, fog);
    m.vertex_colors = vertex_colors;
    m.map = map;
    m.roughness = roughness;
    m.metalness = metalness;
    m.emissive = emissive;
    m.emissive_intensity = emissive_intensity;
    m.transmission = transmission;
    m.thickness = thickness;
    m.ior = ior;
    m.dispersion = dispersion;
    m.attenuation_color = attenuation_color;
    m.attenuation_distance = attenuation_distance;
    return m;
}

// Colors surfaces by their normal direction.
fn MeshNormalMaterial* mesh_normal_material(
    float opacity = 1,
    bool transparent = false,
    Side side = FRONT_SIDE,
    bool wireframe = false
) {
    MeshNormalMaterial* m = mem::new(MeshNormalMaterial);
    m.init_base(NORMAL, 0xffffff, opacity, transparent, side, wireframe, false);
    return m;
}
// Lit by an image of a lit sphere instead of lights (three's MeshMatcapMaterial): `matcap`
// read where the normal points as the viewer sees it, times the color.
fn MeshMatcapMaterial* mesh_matcap_material(
    Color color = 0xffffff,
    Texture* matcap = null,
    float opacity = 1,
    bool transparent = false,
    Side side = FRONT_SIDE,
    bool fog = true,
    Texture* map = null
) {
    MeshMatcapMaterial* m = mem::new(MeshMatcapMaterial);
    m.init_base(MATCAP, color, opacity, transparent, side, false, fog);
    m.matcap = matcap;
    m.map = map;
    return m;
}

// The depth as a color (three's MeshDepthMaterial): its window depth, 0 at the near plane and
// 1 at the far one, packed as `depth_packing` says. Not color managed, as three's.
fn MeshDepthMaterial* mesh_depth_material(
    DepthPacking depth_packing = BASIC_DEPTH_PACKING,
    float opacity = 1,
    bool transparent = false,
    Side side = FRONT_SIDE,
    Texture* map = null
) {
    MeshDepthMaterial* m = mem::new(MeshDepthMaterial);
    m.init_base(DEPTH, 0xffffff, opacity, transparent, side, false, false);
    m.depth_packing = depth_packing;
    m.map = map;
    return m;
}

// Cel shading (three's MeshToonMaterial): lit in steps, from gradient_map, with no highlights.
fn MeshToonMaterial* mesh_toon_material(
    Color color = 0xffffff,
    Texture* gradient_map = null,
    Color emissive = 0x000000,
    float emissive_intensity = 1,
    float opacity = 1,
    bool transparent = false,
    Side side = FRONT_SIDE,
    bool fog = true,
    Texture* map = null
) {
    MeshToonMaterial* m = mem::new(MeshToonMaterial);
    m.init_base(TOON, color, opacity, transparent, side, false, fog);
    m.gradient_map = gradient_map;
    m.emissive = emissive;
    m.emissive_intensity = emissive_intensity;
    m.map = map;
    return m;
}
