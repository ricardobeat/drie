// Shared GPU uniform/SSBO layouts and binding slot constants for runtime-generated material
// and deferred lighting programs. Vertex/depth/shadow structs (`VsParams`, `DepthFsParams`,
// `ShadowVsParams`, `ShadowFsParams`, `InstRow`, `InstIndex`) come from generated/depth_shadow.c3
// and `ClearParams` from generated/clear_rect.c3.
module drie::shaders;
import drie;

// ---- Vertex attribute locations (matching ATTR_DEPTH_*)

const int ATTR_THREE_POSITION       = 0;
const int ATTR_THREE_NORMAL         = 1;
const int ATTR_THREE_UV             = 2;
const int ATTR_THREE_COLOR0         = 3;
const int ATTR_THREE_UV1            = 4;
const int ATTR_THREE_INST0          = 5;
const int ATTR_THREE_INST1          = 6;
const int ATTR_THREE_INST2          = 7;
const int ATTR_THREE_HEIGHT         = 8;
const int ATTR_THREE_INSTANCE_COLOR = 9;
const int ATTR_THREE_TANGENT        = 10;
const int ATTR_THREE_PREV_POSITION  = 11;

// ---- Uniform block, view, and sampler slot indices

const int UB_FS_PARAMS = 1;

const int VIEW_TEX_NORMAL       = 1;
const int VIEW_TEX_ARM          = 2;
const int VIEW_SHADOW_MAP       = 3;
const int VIEW_VSM_MAP          = 3;
const int VIEW_TEX_ENV          = 4;
const int VIEW_SHADOW_ATLAS     = 5;
const int VIEW_VSM_ATLAS        = 5;
const int VIEW_TEX_GI           = 6;
const int VIEW_SSR_COLOR        = 7;
const int VIEW_SSR_DEPTH        = 8;
const int VIEW_SUN_SHADOW       = 9;
const int VIEW_VSM_SUN          = 9;
const int VIEW_DIR_LIGHTS       = 10;
const int VIEW_POINT_LIGHTS     = 11;
const int VIEW_SPOT_LIGHTS      = 12;
const int VIEW_SPOT_SHADOWS     = 13;
const int VIEW_LIGHT_CLUSTERS   = 14;
const int VIEW_LIGHT_INDICES    = 15;
const int VIEW_FOG_VOLUME       = 19;
const int VIEW_SSR_BACK         = 20;
const int VIEW_LIGHT_COOKIES    = 21;
const int VIEW_RC_PROBES        = 22;
const int VIEW_TEX_PROBES       = 23;
const int VIEW_TEX_PARALLAX     = 24;
const int VIEW_TEX_TRANSMISSION = 25;
const int VIEW_TEX_EXT0         = 26;
const int VIEW_TEX_EXT1         = 27;
const int VIEW_TEX_EXT2         = 28;
const int VIEW_TEX_EXT3         = 29;
const int VIEW_TEX_EXT4         = 30;
const int VIEW_TEX_EXT5         = 31;

const int SMP_SHADOW_SMP       = 1;
const int SMP_VSM_SMP          = 1;
const int SMP_ENV_SMP          = 2;
const int SMP_GI_SMP           = 3;
const int SMP_SSR_SMP          = 4;
const int SMP_TRANSMISSION_SMP = 5;

// ---- Fragment uniform block and lighting storage-buffer structs

// Fragment-stage uniform block packed by build_light_params() and draw_item().
struct FsParams @align(16) @packed {
    Vec4 color;
    Vec4 emissive;
    Vec4 specular;
    Vec4 mat_params;
    Vec4 tex_params;
    Vec4 env_params;
    Vec4 env_pos;
    Vec4 env_box_min;
    Vec4 env_box_max;
    Vec4[9] env_sh;
    Vec4 screen_params;
    Vec4 cull_min;
    Vec4 cull_max;
    Vec4 camera_pos;
    Vec4 output_params;
    Vec4 ambient;
    Vec4 hemi_sky;
    Vec4 hemi_ground;
    Vec4[4] gi_min;
    Vec4[4] gi_inv_size;
    Vec4[4] gi_dims;
    Vec4[4] gi_atlas;
    Vec4 fog_color;
    Vec4 fog_params;
    Vec4 fog_box_min;
    Vec4 fog_box_max;
    Vec4 fog_height;
    Vec4 fog_sun_dir;
    Vec4 fog_sun_color;
    Vec4 fog_grid;
    Vec4 fog_slices;
    Vec4 counts;
    Mat4 cluster_view_proj;
    Vec4 cluster_grid;
    Vec4 cluster_depth;
    Mat4[4] dir_shadow_mat;
    Vec4 dir_shadow_params;
    Vec4[4] dir_cascade_rect;
    Vec4[4] dir_cascade_info;
    Vec4 dir_cascade_params;
    Vec4 dir_cascade_fwd;
    Mat4 ssr_view_proj;
    Vec4 ssr_params;
    Vec4 ssr_thickness;
    Vec4 quality_params;
    Mat4 reflector_view_proj;
    Vec4 reflector_params;
    Vec4 probe_info;
    Vec4[4] probe_pos;
    Vec4[4] probe_min;
    Vec4[4] probe_max;
    Vec4 rc_min;
    Vec4 rc_size;
    Vec4 parallax_params;
    Vec4 line_dash;
    Vec4 transmission_params;
    Vec4 transmission_color;
    Vec4 specular_params;
    Vec4 ext_params;
    Vec4[6] map_slots;
    Vec4 clearcoat_params;
    Vec4 sheen_params;
    Vec4 iridescence_params;
    Vec4 tangent_params;
    Vec4 light_map_params;
    Vec4 camera_right;
    Vec4 camera_up;
    Vec4 surface_params;
    Vec4 env_combine;
    Vec4 focus_params;
    Vec4 shadow_ext;
    Vec4 bake_params;
    Vec4[16] uv_rel;
}

// GPU point-light record (also used for RectAreaLight entries and packed LTC table rows).
struct GpuPointLight @align(16) @packed {
    Vec4 pos;
    Vec4 color;
    Vec4 shadow;
    Vec4 params;
}

// GPU shadow-projection record for spot, point-cube, rect-area, and extra directional lights.
struct GpuSpotShadow @align(16) @packed {
    Mat4 mat;
    Vec4 params;
    Vec4 rect;
    Vec4 soft;
}

// GPU spot-light record in VIEW_SPOT_LIGHTS.
struct GpuSpotLight @align(16) @packed {
    Vec4 pos;
    Vec4 dir;
    Vec4 color;
    Vec4 params;
    Vec4 shadow;
    Vec4 map;
    Vec4 map_u;
    Vec4 map_v;
}

// GPU 6-face radiance probe entry in VIEW_RC_PROBES.
struct RcProbe @align(16) @packed {
    Vec4 e;
}

// GPU directional-light record in VIEW_DIR_LIGHTS.
struct GpuDirLight @align(16) @packed {
    Vec4 dir;
    Vec4 color;
    Vec4 shadow;
}

// GPU cluster light-index range (`first`, `npoint`, `nspot`, unused) in VIEW_LIGHT_CLUSTERS.
struct GpuLightCluster @align(16) @packed {
    uint[4] range;
}

// GPU light index entry in VIEW_LIGHT_INDICES.
struct GpuLightIndex @align(4) @packed {
    uint index;
}