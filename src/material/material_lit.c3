// Lit material graph builders: Lambert, Phong, and Standard/Physical shading, direct
// lights, area lights, and environment reflections.
module drie;
import sokol, drie::nodes, drie::shading, std::collections::map, std::io, std::os::env, std::thread;

// ---- the lit materials (lambert, phong, standard)

// A lit material's surface and what its light comes to (the material shader's lighting).
struct Lit @private {
    MaterialKey key;
    Node n;
    Node geo_n;            // the surface's own normal (without the normal map)
    Node frag;             // the pixel's position in the target
    int ltc_fn;            // ltc_evaluate's function (-1: not made yet)
    Parallax parallax;     // where the view ray meets the parallax map (key.parallax)
    FragUv* f;
    Node dpdx;             // how the world position changes from one pixel to the next
    Node dpdy;
    ShadowFns shadow;
    Node v;
    Node world;
    Node rough;
    Node metal;
    Node ao;               // the ambient occlusion of the ARM map (1: none)
    Node diffuse;          // the diffuse color
    Node f0;               // the standard material's reflectance head on, and at a glancing angle
    Node f90;
    Node v_normal;         // the vertex normal, and the vertex stage's tangent (tangent_frame)
    Node tangent;
    // the physical material's layers (where the key has them): a clear coat over the surface
    // (its strength, roughness and normal; what the lights' highlights on it add up to), a
    // sheen of fibers (its color and roughness; its light from the lights and all around), a
    // thin film's rainbow (its strength, and its Fresnel and reflectance head on), highlights
    // drawn out along aniso_t (its strength; aniso_t and across it aniso_b, in the tangent
    // frame's scale)
    Node cc;
    Node cc_rough;
    Node cc_n;
    Node cc_light;
    Node sheen_color;
    Node sheen_rough;
    Node sheen_light;
    Node irid;
    Node irid_fresnel;
    Node irid_f0;
    int irid_fn;
    int probe_fn;          // probe_sample's function (-1: not made yet)
    int ssr_fn;            // screen_reflection's
    int mirror_slot;       // the shared map slot of a Reflector's picture (reflector_slot)
    Refraction refraction;
    Node aniso;
    Node aniso_t;
    Node aniso_b;
    // direct()'s light on the coat and on the sheen, for light_body
    Node coat_d;
    Node sheen_d;
    // the light on the surface: from the lights and all around and given off (all but its
    // highlights), its highlights (none: lambert, toon, basic) and what it gives off itself
    Node lights;
    Node highlights;
    Node glow;
    // its opacity, for glass
    Node alpha;
    // a node material's: emissive_node's glow (none: the material's), env_node (none: the
    // environment as it is); phong's specular color and shininess (specular_node,
    // shininess_node, or the material's)
    Node user_glow;
    Node user_env;
    Node user_cc;          // clearcoat_node, clearcoat_roughness_node (none: the material's)
    Node user_cc_rough;
    Node phong_specular;
    Node shininess;
    // what it comes to: all of the light, and the indirect diffuse part of it
    Node outc;
    Node ind_d;
    Node ind_s;
}

// The light of all the render's lights on the surface: `rgb` the material's color (with its
// map, `map_rgb`), `v_color` the vertex colors.
fn void Lit.shade(&self, FragUv* f, Node rgb, Node map_rgb, Node v_color) {
    MaterialKey key = self.key;
    Node n = self.n;
    self.dpdx = nodes::dfdx(self.world);
    self.dpdy = nodes::dfdy(self.world);
    Node ao = self.ao;
    self.diffuse = rgb;
    if (key.type == (char)MaterialType.STANDARD) {
        // what it reflects head on: 4% for glass's IOR of 1.5, tinted and scaled by the specular
        // color and intensity (KHR_materials_specular); metals reflect their own color
        Node ior_in = fs_in("transmission_params", FLOAT);
        Node ior = nodes::select(ior_in.greater_than(nodes::num(0)), ior_in, 1.5f);
        Node r0 = (ior - 1.0f) / (ior + 1.0f);
        Node sp = fs_in("specular_params");
        Node tint = sp.xyz();
        Node strength = sp.w();
        if (key.slot(MAP_SPECULAR_COLOR)) tint = tint * role_texel(key, f, MAP_SPECULAR_COLOR).xyz();
        if (key.slot(MAP_SPECULAR_INTENSITY)) strength = strength * role_texel(key, f, MAP_SPECULAR_INTENSITY).w();
        Node dielectric = nodes::min(nodes::vec3(r0 * r0) * tint, nodes::vec3(1)) * strength;
        self.f0 = nodes::mix(dielectric, rgb, self.metal);
        self.f90 = nodes::mix(strength, 1, self.metal);
        self.diffuse = rgb * self.metal.one_minus();
        self.layers(f, rgb, dielectric);
    } else if (key.env_reflect) {
        // (the reflection probes on a phong or toon material)
        self.f0 = nodes::mix(nodes::vec3(0.04f), rgb, self.metal);
        self.f90 = nodes::num(1);
    }
    Node diffuse = self.diffuse;

    // the light from all around (ambient and hemisphere lights) and baked into the light map
    Node irradiance = fs_in("ambient", VEC3) + nodes::mix(fs_in("hemi_ground", VEC3), fs_in("hemi_sky", VEC3), n.y() * 0.5f + 0.5f);
    Node env_params = fs_in("env_params");
    if (key.env_light) {
        // the scene's environment: the light from all around n, and its sun, unless a
        // directional light comes from where the sun is: that light is the sun (with its
        // shadow), the picture's is the same light again
        irradiance = irradiance + env_params.y() * env_irradiance(n);
        Node sun = nodes::variable(env_sun(n));
        if (key.dir_lights) {
            Node i = nodes::loop_index(UINT);
            Node dir = node_storage(shaders::VIEW_DIR_LIGHTS, VEC4, i * nodes::uint_(3));
            Node same = nodes::dot(nodes::normalize(dir.xyz()), env_sun_dir()).greater_than(nodes::num(0.996f));
            Node body = nodes::block(nodes::if_then(same, nodes::block(sun.assign(nodes::vec3(0)))));
            Node off = nodes::loop(i, nodes::uint_(0), nodes::convert(fs_in("counts", FLOAT), UINT), body);
            irradiance = irradiance + env_params.y() * nodes::seq(nodes::block(sun, off), sun);
        } else {
            irradiance = irradiance + env_params.y() * sun;
        }
    }
    if (key.gi) {
        // the scene's irradiance volumes: added to the light from all around, or in its place
        // where they cover the point (the first's gi_min.w 2)
        Node[2] gi = gi_irradiance(n, self.world);
        irradiance = nodes::select(fs_in("gi_min", FLOAT, 3).greater_than(nodes::num(1.5f)),
                                   irradiance * gi[1].one_minus() + gi[0], irradiance + gi[0]);
    }
    if (key.rc) {
        // radiance cascades: all the indirect light inside their box
        Node[2] rc = rc_irradiance(n, self.world);
        irradiance = nodes::mix(irradiance, rc[0], rc[1]);
    }
    if (key.slot(MAP_LIGHT)) irradiance = irradiance + role_texel(key, f, MAP_LIGHT).xyz() * fs_in("light_map_params", FLOAT);
    Node outc0 = irradiance * diffuse * shading::RECIPROCAL_PI * ao;
    Node sheen_ind;
    if (key.sheen) {
        // the sheen's light from all around, and what the fibers take of it
        Node albedo = sheen_albedo(n, self.v, self.sheen_rough);
        sheen_ind = irradiance * self.sheen_color * albedo * shading::RECIPROCAL_PI;
        outc0 = outc0 * (max3(self.sheen_color) * albedo).one_minus();
    }
    self.ind_d = outc0;

    // the lights, each added to outc and to spec, the highlights
    Node outc = nodes::variable(outc0);
    Node spec = nodes::variable(nodes::vec3(0));
    Node[20] stmts;
    int ns;
    stmts[ns++] = drie::node_eval(n);
    stmts[ns++] = drie::node_eval(self.v);
    stmts[ns++] = drie::node_eval(diffuse);
    stmts[ns++] = outc;
    stmts[ns++] = spec;
    if (key.clearcoat) {
        self.cc_light = nodes::variable(nodes::vec3(0));
        stmts[ns++] = self.cc_light;
    }
    if (key.sheen) {
        self.sheen_light = nodes::variable(sheen_ind);
        stmts[ns++] = self.sheen_light;
    }
    if (key.dir_lights) {
        Node i = nodes::loop_index(UINT);
        Node at = i * nodes::uint_(3);
        Node dir = node_storage(shaders::VIEW_DIR_LIGHTS, VEC4, at);
        Node color = node_storage(shaders::VIEW_DIR_LIGHTS, VEC4, at + nodes::uint_(1));
        Node shadow = node_storage(shaders::VIEW_DIR_LIGHTS, VEC4, at + nodes::uint_(2));
        Node l = nodes::normalize(dir.xyz());
        Node radiance = color.xyz();
        Node[6] pre;
        int np;
        Node skip;
        if (key.sun_shadow || key.dir_tiles || key.parallax) {
            Node world = self.world;
            Node lit = nodes::variable(nodes::num(1));
            pre[np++] = drie::node_eval(l);
            pre[np++] = lit;
            if (key.sun_shadow) {
                Node sun = nodes::convert(fs_in("dir_shadow_params", FLOAT), INT).equal(nodes::convert(i, INT) + nodes::int_(1));
                pre[np++] = nodes::if_then(sun, nodes::block(lit.assign(self.shadow.sun(n, l, world, self.frag))));
            }
            if (key.dir_tiles) {
                // the other shadowed ones: a tile of shadow_atlas each
                Node dsi = nodes::convert(shadow.x(), INT);
                Node tile = nodes::mix(1, self.shadow.dir_tile(dsi, n, l, world, self.frag), shadow.y());
                pre[np++] = nodes::if_then(dsi.greater_than_equal(nodes::int_(0)), nodes::block(lit.assign(tile)));
            }
            if (key.parallax) {
                // parallax_map: the bumps' own shadows, toward the first directional light
                Node bumps = parallax_shadow(&self.parallax, self.f, self.geo_n, l);
                pre[np++] = nodes::if_then(i.equal(nodes::uint_(0)), nodes::block(lit.assign(lit * bumps)));
            }
            skip = lit.less_than_equal(nodes::num(0));
            radiance = radiance * lit;
        }
        Node body = self.light_body(shadow, self.direct(l, radiance), outc, spec, skip, pre[:np]);
        stmts[ns++] = nodes::loop(i, nodes::uint_(0), nodes::convert(fs_in("counts", FLOAT), UINT), body);
    }
    if (key.debug_cascades) {
        // renderer.debug_cascades: red, green, blue, yellow
        Node k = sun_cascade(sun_depth(self.world));
        Node tint = nodes::select(k.equal(nodes::int_(0)), nodes::vec3(1, 0, 0),
                    nodes::select(k.equal(nodes::int_(1)), nodes::vec3(0, 1, 0),
                    nodes::select(k.equal(nodes::int_(2)), nodes::vec3(0, 0, 1), nodes::vec3(1, 1, 0))));
        stmts[ns++] = nodes::if_then(k.greater_than_equal(nodes::int_(0)), nodes::block(outc.assign(nodes::mix(outc, tint, 0.5f))));
    }
    // where a light with a size is mirrored from (light_toward)
    Node mirror_r = nodes::reflect(self.v.negate(), n);
    if (key.point_lights || key.spot_lights) {
        // the point and spot lights that reach the fragment's cluster
        // (baking: all of them, a light map's texel having no cluster)
        Node range;
        Node first;
        Node npoint;
        Node nspot;
        if (key.bake) {
            npoint = nodes::convert(fs_in("counts", FLOAT, 1), UINT);
            nspot = nodes::convert(fs_in("counts", FLOAT, 2), UINT);
        } else {
            range = node_storage(shaders::VIEW_LIGHT_CLUSTERS, UVEC4, nodes::convert(cluster_of(self.world), UINT));
            first = range.x();
            npoint = range.y();
            nspot = range.z();
            stmts[ns++] = drie::node_eval(range);
        }
        if (key.point_lights) {
            Node k = nodes::loop_index(UINT);
            Node at = (key.bake ? k : node_storage(shaders::VIEW_LIGHT_INDICES, UINT, first + k)) * nodes::uint_(4);
            Node pos = node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at);
            Node color = node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at + nodes::uint_(1));
            Node shadow = node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at + nodes::uint_(2));
            Node params = node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at + nodes::uint_(3));
            Node lv = pos.xyz() - self.world;
            Node d = nodes::length(lv);
            // a light source with a size: falloff softens up close instead of blowing out
            Node radius = params.x();
            Node seg = params.swizzle("yzw");
            Node l = lv / d;
            Node d_soft = nodes::sqrt(d * d + radius * radius);
            Node radiance = color.xyz() * distance_attenuation(d_soft, pos.w(), color.w());
            Sized size;
            if (key.sized) {
                size = light_toward(mirror_r, lv, seg, radius, key.tubes);
                size.on = (radius + nodes::abs(seg.x()) + nodes::abs(seg.y()) + nodes::abs(seg.z())).greater_than(nodes::num(0));
            }
            if (key.tubes) {
                // a tube (PointLight.length): the light of the whole segment, its intensity
                // spread along it (Karis' eq. 16, the radius softening it as a ball's falloff:
                // d^2 + radius^2 at length 0), faded out by the distance to its nearest point
                Node tube = nodes::dot(seg, seg).greater_than(nodes::num(0));
                Node l0 = lv - seg;
                Node l1 = lv + seg;
                Node d0 = nodes::length(l0);
                Node d1 = nodes::length(l1);
                Node ld = seg * 2.0f;
                Node near = l0 + ld * nodes::clamp(nodes::dot(l0, ld).negate() / nodes::dot(ld, ld), 0, 1);
                Node dn = nodes::length(near);
                Node across = nodes::clamp(nodes::dot(n, l0) * 0.5f / d0 + nodes::dot(n, l1) * 0.5f / d1, 0, 1) * 2.0f;
                Node nol = across / nodes::max(d0 * d1 + nodes::dot(l0, l1) + radius * radius * 2.0f, 1e-4f);
                size.nol = nodes::select(tube, nol, -1.0f);
                l = nodes::select(tube, near / nodes::max(dn, 1e-6f), l);
                radiance = nodes::select(tube, color.xyz() * distance_window(dn, pos.w()), radiance);
            }
            if (key.fog) radiance = radiance * shading::fog_transmittance(key, shading::fog_distance(key, pos.xyz(), self.world));
            Node[2] pre;
            int np;
            if (key.point_shadows) {
                Node shade = nodes::variable(nodes::num(1));
                Node si = nodes::convert(shadow.x(), INT);
                Node lit = nodes::mix(1, self.shadow.point(pos.xyz(), si, self.geo_n, self.world, self.dpdx, self.dpdy, self.frag), shadow.y());
                pre[np++] = shade;
                pre[np++] = nodes::if_then(si.greater_than_equal(nodes::int_(0)), nodes::block(shade.assign(lit)));
                radiance = radiance * shade;
            }
            Node body = self.light_body(shadow, self.direct(l, radiance, &size), outc, spec, { }, pre[:np]);
            stmts[ns++] = nodes::loop(k, nodes::uint_(0), npoint, body);
        }
        if (key.spot_lights) {
            Node k = nodes::loop_index(UINT);
            Node at = (key.bake ? k : node_storage(shaders::VIEW_LIGHT_INDICES, UINT, first + npoint + k)) * nodes::uint_(8);
            Node pos = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at);
            Node dir = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(1));
            Node color = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(2));
            Node params = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(3));
            Node shadow = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(4));
            Node lv = pos.xyz() - self.world;
            Node d = nodes::length(lv);
            Node l = lv / d;
            Node cone = nodes::smoothstep(dir.w(), params.x(), nodes::dot(l.negate(), dir.xyz()));
            Node d_soft = nodes::sqrt(d * d + params.y() * params.y());
            Node radiance = color.xyz() * cone * distance_attenuation(d_soft, pos.w(), color.w());
            Sized size;
            if (key.sized) {
                size = light_toward(mirror_r, lv, { }, params.z(), false);
                size.on = params.z().greater_than(nodes::num(0));
            }
            if (key.fog) radiance = radiance * shading::fog_transmittance(key, shading::fog_distance(key, pos.xyz(), self.world));
            // out of the light's reach, or of its cone
            Node away = nodes::logical_or(
                nodes::logical_and(pos.w().greater_than(nodes::num(0)), d.greater_than(pos.w())),
                cone.less_than_equal(nodes::num(0)));
            Node[4] pre;
            int np;
            {
                // SpotLight.map (shader_core.glsl's spot_cookie): the picture it projects, from
                // light_cookies
                Node map = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(5));
                Node map_u = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(6));
                Node map_v = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(7));
                Node w = l.negate();
                Node z = nodes::max(nodes::dot(w, dir.xyz()), 1e-4f);
                Node av = nodes::vec3(map_u.w(), map_v.x(), map_v.y());
                Node uv = nodes::vec2(nodes::dot(w, map_u.xyz()), nodes::dot(w, av)) / z * 0.5f + 0.5f;
                Node off = nodes::abs(uv - 0.5f);
                Node inside = nodes::max(off.x(), off.y()).less_than_equal(nodes::num(0.5f));
                Node pixel = nodes::max(nodes::length(self.dpdx), nodes::length(self.dpdy));
                Node lod = nodes::max(nodes::log2(nodes::max(pixel * 0.5f * nodes::length(map_u.xyz()) / (z * d), 1e-8f)) + map_v.z(), 0.0f);
                Node tap = map.xy() + nodes::vec2(uv.x(), uv.y().negate() + 1.0f) * map.swizzle("zw");
                Node texel = node_sample(LEVEL, shaders::VIEW_LIGHT_COOKIES, shaders::SMP_SSR_SMP, TEX_2D, tap, lod).xyz();
                Node cookie = nodes::variable(nodes::vec3(1, 1, 1));
                pre[np++] = cookie;
                pre[np++] = nodes::if_then(map.z().greater_than(nodes::num(0)),
                    nodes::block(cookie.assign(nodes::select(inside, texel, nodes::vec3(0, 0, 0)))));
                radiance = radiance * cookie;
            }
            if (key.spot_shadows) {
                Node shade = nodes::variable(nodes::num(1));
                Node si = nodes::convert(shadow.x(), INT);
                Node lit = nodes::mix(1, self.shadow.light(pos.xyz(), si, self.geo_n, self.world, self.dpdx, self.dpdy, self.frag), shadow.y());
                pre[np++] = shade;
                pre[np++] = nodes::if_then(si.greater_than_equal(nodes::int_(0)), nodes::block(shade.assign(lit)));
                radiance = radiance * shade;
            }
            Node body = self.light_body(shadow, self.direct(l, radiance, &size), outc, spec, away, pre[:np]);
            stmts[ns++] = nodes::loop(k, nodes::uint_(0), nspot, body);
        }
    }
    if (key.rect_lights) stmts[ns++] = self.rect_lights(outc, spec);
    Node lit = nodes::seq(drie::node_block(stmts[:ns]), outc);
    Node highlights = spec;
    // three's specularMap: the highlights' strength
    if (key.type == (char)MaterialType.PHONG && key.slot(MAP_SPECULAR)) highlights = highlights * role_texel(key, f, MAP_SPECULAR).x();
    bool shiny = key.type != (char)MaterialType.LAMBERT && key.type != (char)MaterialType.TOON;
    if (shiny) self.highlights = highlights * ao;
    // what it gives off itself: Material.map_emissive (the color map glows too), three's emissiveMap
    Node glow = fs_in("emissive", VEC3);
    if (key.map_emissive && map_rgb.id) glow = glow * map_rgb;
    if (key.slot(MAP_EMISSIVE)) glow = glow * role_texel(key, f, MAP_EMISSIVE).xyz();
    if (self.user_glow.id) glow = self.user_glow;
    self.glow = glow;
    Node result = lit + glow;
    // light baked into the vertex colors, as if from fixed lamps
    if (key.baked) {
        Node b = diffuse * v_color * fs_in("emissive", FLOAT, 3) * ao;
        result = result + b;
        self.ind_d = self.ind_d + b;
    }
    self.lights = result;
    self.finish();
}

// What comes over the light on the surface (self.lights, its highlights and the light it
// gives off), for every material type: glass's reflections; or the environment's reflection,
// the mirror images over it, the light through a transmission material and the physical
// material's sheen and clear coat.
fn void Lit.finish(&self) {
    MaterialKey key = self.key;
    Node n = self.n;
    Node ao = self.ao;
    FragUv* f = self.f;
    if (key.glass) {
        // glass: reflections stay visible however clear the pane is
        Node v = self.v;
        Node fresnel = nodes::pow(nodes::clamp(nodes::dot(n, v), 0, 1).one_minus(), 5) * 0.96f + 0.04f;
        Node r = nodes::reflect(v.negate(), n);
        Node sky;
        if (key.env_reflect) {
            sky = env_sample(key, &self.probe_fn, r, self.rough, self.world, self.frag, n);
        } else {
            sky = nodes::mix(fs_in("hemi_ground", VEC3), fs_in("hemi_sky", VEC3), r.y() * 0.5f + 0.5f) + fs_in("ambient", VEC3);
        }
        sky = self.env_node(sky, r, self.rough);
        Node spec = sky * fresnel;
        if (self.highlights.id) spec = self.highlights + spec;
        Node a = nodes::clamp(self.alpha + fresnel + max3(spec), 0, 1);
        // blended with src-alpha: pre-divided so that outc * alpha + spec comes out
        self.outc = (self.lights * self.alpha + spec) / nodes::max(a, 1e-4f);
        self.alpha = a;
        self.ind_s = nodes::vec3(0);
        return;
    }
    self.ind_s = nodes::vec3(0);
    Node env_term;
    if (key.env_reflect) {
        // reflections of the surroundings, fading as the surface gets rougher
        Node v = self.v;
        Node rough = self.rough;
        Node nov = nodes::clamp(nodes::dot(n, v), 0, 1);
        Node f0 = self.f0;
        if (key.iridescence) f0 = nodes::select(self.irid.greater_than(nodes::num(0)), nodes::mix(f0, self.irid_f0, self.irid), f0);
        Node fr = f0 + (nodes::max(nodes::vec3(rough.one_minus() * self.f90), f0) - f0) * nodes::pow(nov.one_minus(), 5);
        Node rn = n;
        if (key.anisotropy) {
            // the normal bent toward the stretched highlight (three's getIBLAnisotropyRadiance,
            // Filament's)
            Node bent = nodes::normalize(nodes::cross(nodes::cross(self.aniso_b, v), self.aniso_b));
            Node k = (self.aniso * rough.one_minus()).one_minus();
            k = k * k;
            rn = nodes::select(self.aniso.greater_than(nodes::num(0)), nodes::normalize(nodes::mix(bent, n, k * k)), n);
        }
        Node r = dominant_reflection(nodes::reflect(v.negate(), rn), rn, rough);
        env_term = self.env_node(env_sample(key, &self.probe_fn, r, rough, self.world, self.frag, n), r, rough) * fr;
        if (key.sheen) env_term = env_term * (max3(self.sheen_color) * sheen_albedo(n, v, self.sheen_rough)).one_minus();
        self.ind_s = env_term;
    }
    Node mirror_term;
    if (key.ssr || key.reflector) {
        // the surface's sheen: a mirror image, strongest at a glancing angle (far off, where a
        // pixel covers many bumps of the normal map, each would throw the image somewhere
        // else: blotches. There the surface mirrors as if flat)
        Node v = self.v;
        Node bumps = nodes::smoothstep(1.5f, 5.0f, nodes::length(fs_in("camera_pos", VEC3) - self.world)).one_minus() * 0.5f;
        Node rn = nodes::normalize(nodes::mix(self.geo_n, n, bumps));
        Node fresnel = nodes::pow(nodes::clamp(nodes::dot(rn, v), 0, 1).one_minus(), 5) * 0.96f + 0.04f;
        Node mirrored;
        Node found;
        Node strength;
        if (key.reflector) {
            // the planar reflection: the picture of the mirrored scene where the ray points
            Node seen = planar_reflection(rn, v, self.rough, self.world, self.frag, self.mirror_slot);
            mirrored = seen.xyz();
            found = seen.w() * fs_in("reflector_params", FLOAT);
            strength = nodes::num(1);
        }
        if (key.ssr) {
            // what is on screen near it, where a ray finds that: sharper, at full size
            Node near = screen_reflection(&self.ssr_fn, rn, self.geo_n, v, self.rough, self.world, self.frag);
            Node ssr = fs_in("ssr_params", FLOAT);
            if (key.reflector) {
                mirrored = nodes::select(found.greater_than(nodes::num(0)), nodes::mix(mirrored, near.xyz(), near.w()), near.xyz());
                found = nodes::max(found, near.w());
                strength = nodes::max(strength, ssr);
            } else {
                mirrored = near.xyz();
                found = near.w();
                strength = ssr;
            }
        }
        // where something was found, it takes the place of the surroundings' map
        mirror_term = mirrored * (found * fresnel * strength * ao);
        if (env_term.id) env_term = env_term * nodes::clamp(found * strength, 0, 1).one_minus();
        self.ind_s = env_term.id ? env_term + mirror_term : mirror_term;
    }
    Node result = self.lights;
    if (env_term.id) result = result + env_term;
    if (mirror_term.id) result = result + mirror_term;
    self.outc = self.highlights.id ? result + self.highlights : result;
    if (key.transmission) {
        // what is behind the surface, refracted through the volume, takes the place of its
        // diffuse light; highlights, reflections and the light it gives off stay
        Node amount = fs_in("transmission_color", FLOAT, 3);
        if (key.slot(MAP_TRANSMISSION)) amount = amount * role_texel(key, f, MAP_TRANSMISSION).x();
        Node thickness = nodes::max(fs_in("transmission_params", FLOAT, 1), 0);
        if (key.slot(MAP_THICKNESS)) thickness = thickness * role_texel(key, f, MAP_THICKNESS).y();
        Node refracted = self.refraction.light(n, self.v, self.world, self.rough, self.diffuse, thickness, self.f0, self.f90);
        Node kept = self.highlights + self.glow;
        if (env_term.id) kept = kept + env_term;
        if (mirror_term.id) kept = kept + mirror_term;
        self.outc = nodes::mix(nodes::max(self.outc - kept, nodes::vec3(0)), refracted, amount) + kept;
    }
    // the physical material's layers over it all: the sheen, then the clear coat, which lets
    // through what it does not reflect (three's)
    if (key.sheen) {
        self.outc = self.outc + self.sheen_light;
        self.ind_s = self.ind_s + self.sheen_light;
    }
    if (key.clearcoat) {
        Node cn = self.cc_n;
        Node v = self.v;
        Node coat = self.cc_light;
        if (key.env_reflect) {
            Node r = dominant_reflection(nodes::reflect(v.negate(), cn), cn, self.cc_rough);
            Node env = self.env_node(env_sample(key, &self.probe_fn, r, self.cc_rough, self.world, self.frag, cn), r, self.cc_rough);
            Node reflected = env * environment_brdf(cn, v, nodes::vec3(0.04f), nodes::num(1), self.cc_rough);
            coat = coat + reflected;
            self.ind_s = self.ind_s + reflected * self.cc;
        }
        Node fc = schlick(nodes::num(0.04f), nodes::num(1), nodes::clamp(nodes::dot(cn, v), 0, 1));
        self.outc = self.outc * (self.cc * fc).one_minus() + coat * self.cc;
    }
}

// The environment's reflection `env` in direction `dir` at roughness `rough`, or a node
// material's env_node in its place (node_env), which reads them as ENV_DIRECTION and
// ENV_ROUGHNESS.
fn Node Lit.env_node(&self, Node env, Node dir, Node rough) {
    if (!self.user_env.id) return env;
    return bound(nodes::convert(self.user_env, VEC3), { rebind(ENV_DIRECTION, dir), rebind(ENV_ROUGHNESS, rough),
                                                        rebind(NORMAL_WORLD, self.n), rebind(NORMAL_GEOMETRY, self.geo_n),
                                                        rebind(ENV_COLOR, env) });
}

// The physical material's layers (key.clearcoat, sheen, iridescence, anisotropy) from its
// parameters and maps; `base` its color, `dielectric` its reflectance head on as a dielectric.
fn void Lit.layers(&self, FragUv* f, Node base, Node dielectric) {
    MaterialKey key = self.key;
    Node geo_n = self.geo_n;
    if (key.clearcoat) {
        // the clear coat: its strength and roughness, and its normal (the surface's own, under
        // the normal map, or from its own map)
        Node p = fs_in("clearcoat_params");
        Node cc = p.x();
        if (key.slot(MAP_CLEARCOAT)) cc = cc * role_texel(key, f, MAP_CLEARCOAT).x();
        // (a node material's clearcoat_node and clearcoat_roughness_node in their place)
        if (self.user_cc.id) cc = bound(nodes::convert(self.user_cc, FLOAT), { rebind(NORMAL_WORLD, self.n), rebind(NORMAL_GEOMETRY, geo_n) });
        self.cc = nodes::clamp(cc, 0, 1);
        Node rough = p.y();
        if (key.slot(MAP_CLEARCOAT_ROUGHNESS)) rough = rough * role_texel(key, f, MAP_CLEARCOAT_ROUGHNESS).y();
        if (self.user_cc_rough.id) {
            rough = bound(nodes::convert(self.user_cc_rough, FLOAT), { rebind(NORMAL_WORLD, self.n), rebind(NORMAL_GEOMETRY, geo_n) });
        }
        self.cc_rough = nodes::clamp(rough, 0.045f, 1);
        self.cc_n = geo_n;
        if (key.slot(MAP_CLEARCOAT_NORMAL)) {
            Node t = role_texel(key, f, MAP_CLEARCOAT_NORMAL).xyz() * 2.0f - 1.0f;
            t = nodes::vec3(t.xy() * p.swizzle("zw"), t.z());
            int u = key.slot(MAP_CLEARCOAT_NORMAL) + 1;
            Node[2] tb = tangent_frame(key, geo_n, self.v_normal, self.world, self.tangent, shading::slot_d(f, u, false), shading::slot_d(f, u, true));
            self.cc_n = nodes::normalize(tb[0] * t.x() + tb[1] * t.y() + geo_n * t.z());
        }
    }
    if (key.sheen) {
        // the sheen (0 roughness: none)
        Node p = fs_in("sheen_params");
        self.sheen_color = p.xyz();
        self.sheen_rough = nodes::clamp(p.w(), 1e-4f, 1);
        if (key.slot(MAP_SHEEN_COLOR)) self.sheen_color = self.sheen_color * role_texel(key, f, MAP_SHEEN_COLOR).xyz();
        if (key.slot(MAP_SHEEN_ROUGHNESS)) self.sheen_rough = self.sheen_rough * role_texel(key, f, MAP_SHEEN_ROUGHNESS).w();
    }
    if (key.iridescence) {
        // iridescence (three's): the film's reflectance over a dielectric and over a metal
        Node p = fs_in("iridescence_params");
        Node irid = p.x();
        if (key.slot(MAP_IRIDESCENCE)) irid = irid * role_texel(key, f, MAP_IRIDESCENCE).x();
        Node film = p.w();
        if (key.slot(MAP_IRIDESCENCE_THICKNESS)) film = nodes::mix(p.z(), p.w(), role_texel(key, f, MAP_IRIDESCENCE_THICKNESS).y());
        self.irid = nodes::select(film.equal(nodes::num(0)), 0.0f, nodes::clamp(irid, 0, 1));
        Node nov = nodes::clamp(nodes::dot(self.n, self.v), 0, 1);
        Node over_dielectric = eval_iridescence(&self.irid_fn, nodes::num(1), p.y(), nov, film, dielectric);
        Node over_metal = eval_iridescence(&self.irid_fn, nodes::num(1), p.y(), nov, film, base);
        self.irid_fresnel = nodes::mix(over_dielectric, over_metal, self.metal);
        self.irid_f0 = nodes::mix(schlick_to_f0(over_dielectric, nodes::num(1), nov), schlick_to_f0(over_metal, nodes::num(1), nov), self.metal);
    }
    if (key.anisotropy) {
        // anisotropy (three's): highlights drawn out along a direction of the surface, as on
        // brushed metal. The direction in the tangent frame, turned by the map's (rg, around its
        // middle) and scaled by its b
        Node dir = fs_in("tangent_params", VEC2, 2);
        if (key.slot(MAP_ANISOTROPY)) {
            Node polar = role_texel(key, f, MAP_ANISOTROPY).xyz();
            Node turn = polar.xy() * 2.0f - 1.0f;
            turn = turn * (polar.z() / nodes::max(nodes::length(turn), 1e-6f));
            dir = nodes::vec2(dir.x() * turn.x() - dir.y() * turn.y(), dir.y() * turn.x() + dir.x() * turn.y());
        }
        Node amount = nodes::length(dir);
        dir = nodes::select(amount.greater_than(nodes::num(0)), dir / amount, nodes::vec2(1, 0));
        self.aniso = nodes::min(amount, 1);
        // the frame: the normal map's uv, or the clear coat's normal map's, or the color map's
        Node duv1 = f.dx;
        Node duv2 = f.dy;
        if (key.normal_map) {
            duv1 = shading::slot_d(f, 0, false);
            duv2 = shading::slot_d(f, 0, true);
        } else if (key.slot(MAP_CLEARCOAT_NORMAL)) {
            int u = key.slot(MAP_CLEARCOAT_NORMAL) + 1;
            duv1 = shading::slot_d(f, u, false);
            duv2 = shading::slot_d(f, u, true);
        }
        Node[2] tb = tangent_frame(key, geo_n, self.v_normal, self.world, self.tangent, duv1, duv2);
        self.aniso_t = tb[0] * dir.x() + tb[1] * dir.y();
        self.aniso_b = tb[1] * dir.x() - tb[0] * dir.y();
    }
}

// A light's loop body: skipped where the light is in none of the material's channels; then the
// statements `pre` (its shadow), and skipped where `skip` holds, else its diffuse light `d[0]`
// added to `outc` and its highlight `d[1]` to `spec`.
fn Node Lit.light_body(&self, Node shadow, Node[2] d, Node outc, Node spec, Node skip = { }, Node[] pre = { }) {
    Node[12] st;
    int n;
    if (self.key.channels) {
        // (lights_material: a light of none of the material's channels)
        Node taken = nodes::convert(fs_in("surface_params", FLOAT, 3), UINT);
        Node none = nodes::bit_and(taken, nodes::convert(shadow.z(), UINT)).equal(nodes::uint_(0));
        st[n++] = nodes::if_then(none, nodes::block(nodes::continue_()));
    }
    foreach (p : pre) st[n++] = p;
    if (skip.id) st[n++] = nodes::if_then(skip, nodes::block(nodes::continue_()));
    st[n++] = outc.assign(outc + d[0]);
    if (d[1].id) st[n++] = spec.assign(spec + d[1]);
    // (the physical material's layers: what direct() left for the coat and the sheen)
    if (self.coat_d.id) st[n++] = self.cc_light.assign(self.cc_light + self.coat_d);
    if (self.sheen_d.id) st[n++] = self.sheen_light.assign(self.sheen_light + self.sheen_d);
    self.coat_d = { };
    self.sheen_d = { };
    return drie::node_block(st[:n]);
}

// RectAreaLight, as three's (lights_physical_pars_fragment: LTC_Uv, LTC_Evaluate and
// RE_Direct_RectArea_Physical), with linearly transformed cosines (Heitz, Dupuy, Hill and
// Neubelt 2016). The lights are in point_lights after the point lights: pos = centre + half
// width's x, color = color * intensity + half width's y, shadow.z = its channels, params = half
// width's z + half height; and after them the two LTC tables (clusters.c3's ltc_entries), a
// texel an entry (pos = LTC_1's, color = LTC_2's), read here with the bilinear filter three's
// textures have. The loop adds their diffuse light to `outc` and their highlights to `spec`.
fn Node Lit.rect_lights(&self, Node outc, Node spec) {
    MaterialKey key = self.key;
    Node n = self.n;
    Node v = self.v;
    Node counts = fs_in("counts");
    Node npoint = nodes::convert(counts.y(), INT);
    Node nrect = nodes::convert(counts.w(), INT);
    Node[2] t = ltc_tables(npoint + nrect, n, v, nodes::max(self.rough, 0.0525f));
    Node t1 = t[0];
    Node t2 = t[1];
    Node[3] inv_cols = { nodes::vec3(t1.x(), 0, t1.y()), nodes::vec3(0, 1, 0), nodes::vec3(t1.z(), 0, t1.w()) };
    Node m_inv = node_matrix(&inv_cols, MAT3);
    Node[3] id_cols = { nodes::vec3(1, 0, 0), nodes::vec3(0, 1, 0), nodes::vec3(0, 0, 1) };
    Node ident = node_matrix(&id_cols, MAT3);
    Node fresnel = self.f0 * t2.x() + (nodes::vec3(self.f90) - self.f0) * t2.y();
    Node cc_inv;
    Node cc_fresnel;
    if (key.clearcoat) {
        Node[2] c = ltc_tables(npoint + nrect, self.cc_n, v, nodes::max(self.cc_rough, 0.0525f));
        Node[3] cols = { nodes::vec3(c[0].x(), 0, c[0].y()), nodes::vec3(0, 1, 0), nodes::vec3(c[0].z(), 0, c[0].w()) };
        cc_inv = node_matrix(&cols, MAT3);
        cc_fresnel = c[1].x() * 0.04f + c[1].y() * 0.96f;
    }

    Node k = nodes::loop_index();
    Node at = (npoint + k) * nodes::int_(4);
    Node pos = point_light_row(at);
    Node color = point_light_row(at + nodes::int_(1));
    Node shadow = point_light_row(at + nodes::int_(2));
    Node params = point_light_row(at + nodes::int_(3));
    Node hw = nodes::vec3(pos.w(), color.w(), params.x());
    Node hh = params.swizzle("yzw");
    Node to = pos.xyz() - self.world;
    // behind the light (it shines from its front only)
    Node behind = nodes::dot(nodes::cross(hw, hh), to).less_than(nodes::num(0));
    Node[4] q = { to + hw - hh, to - hw - hh, to - hw + hh, to + hw + hh };
    Node c = color.xyz();
    if (key.fog) c = c * shading::fog_transmittance(key, shading::fog_distance(key, pos.xyz(), self.world));
    // RectAreaLight.shadow, cast from its centre (not looked up for a point behind the light)
    Node[3] pre;
    int np;
    if (key.point_shadows) {
        Node shade = nodes::variable(nodes::num(1));
        Node si = nodes::convert(shadow.x(), INT);
        Node lit = nodes::mix(1, self.shadow.rect(pos.xyz(), hw, hh, si, self.geo_n, self.world, self.dpdx, self.dpdy, self.frag), shadow.y());
        pre[np++] = nodes::if_then(behind, nodes::block(nodes::continue_()));
        pre[np++] = shade;
        pre[np++] = nodes::if_then(si.greater_than_equal(nodes::int_(0)), nodes::block(shade.assign(lit)));
        c = c * shade;
        behind = { };
    }
    Node[2] d = { c * self.diffuse * self.ltc_evaluate(n, v, ident, &q), c * fresnel * self.ltc_evaluate(n, v, m_inv, &q) };
    if (key.clearcoat) self.coat_d = c * cc_fresnel * self.ltc_evaluate(self.cc_n, v, cc_inv, &q);
    Node body = self.light_body(shadow, d, outc, spec, behind, pre[:np]);
    return nodes::seq(nodes::block(drie::node_eval(t1), drie::node_eval(t2), nodes::loop(k, 0, nrect, body)), outc);
}

// The two LTC tables at roughness `rough` and the angle of v to n (three's LTC_Uv: roughness
// across, sqrt(1 - n.v) down, at the texels' centres), read bilinearly from point_lights at
// `base`.
fn Node[2] ltc_tables(Node base, Node n, Node v, Node rough) @local {
    Node uv = nodes::vec2(rough, nodes::sqrt(nodes::clamp(nodes::dot(n, v), 0, 1).one_minus())) * (63.0f / 64.0f) + 0.5f / 64.0f;
    Node p = uv * 64.0f - 0.5f;
    Node fr = nodes::fract(p);
    Node fl = nodes::floor(p);
    Node i0x = nodes::clamp(nodes::convert(fl.x(), INT), nodes::int_(0), nodes::int_(63));
    Node i0y = nodes::clamp(nodes::convert(fl.y(), INT), nodes::int_(0), nodes::int_(63));
    Node i1x = nodes::min(i0x + nodes::int_(1), nodes::int_(63));
    Node i1y = nodes::min(i0y + nodes::int_(1), nodes::int_(63));
    Node ta = (base + i0y * nodes::int_(64) + i0x) * nodes::int_(4);
    Node tb = (base + i0y * nodes::int_(64) + i1x) * nodes::int_(4);
    Node tc = (base + i1y * nodes::int_(64) + i0x) * nodes::int_(4);
    Node td = (base + i1y * nodes::int_(64) + i1x) * nodes::int_(4);
    Node[2] t;
    for (int e = 0; e < 2; e++) {
        Node one = nodes::int_(e);
        Node top = nodes::mix(point_light_row(ta + one), point_light_row(tb + one), fr.x());
        Node bottom = nodes::mix(point_light_row(tc + one), point_light_row(td + one), fr.x());
        t[e] = nodes::mix(top, bottom, fr.y());
    }
    return t;
}

fn Node point_light_row(Node index) @local => node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, index);

// How much of the cosine lobe, transformed by m_inv, the rectangle (corners q, from the
// surface) covers: three's LTC_Evaluate.
fn Node Lit.ltc_evaluate(&self, Node n, Node v, Node m_inv, Node[4]* q) {
    if (self.ltc_fn < 0) {
        NodeType[7] types = { VEC3, VEC3, MAT3, VEC3, VEC3, VEC3, VEC3 };
        int f = node_func("ltc_evaluate", FLOAT, &types);
        Node pn = node_param(f, 0);
        Node pv = node_param(f, 1);
        Node pm = node_param(f, 2);
        Node t1 = nodes::normalize(pv - pn * nodes::dot(pv, pn));
        Node t2 = nodes::cross(pn, t1).negate();
        // (transpose(mat3(t1, t2, n)))
        Node[3] cols = { nodes::vec3(t1.x(), t2.x(), pn.x()), nodes::vec3(t1.y(), t2.y(), pn.y()), nodes::vec3(t1.z(), t2.z(), pn.z()) };
        Node m = pm * node_matrix(&cols, MAT3);
        Node[4] c;
        for (int i = 0; i < 4; i++) c[i] = nodes::normalize(m * node_param(f, 3 + i));
        Node e = ltc_edge(c[0], c[1]) + ltc_edge(c[1], c[2]) + ltc_edge(c[2], c[3]) + ltc_edge(c[3], c[0]);
        // three's LTC_ClippedSphereFormFactor
        Node l = nodes::length(e);
        node_func_body(f, nodes::max((l * l + e.z()) / (l + 1.0f), 0));
        self.ltc_fn = f;
    }
    Node[7] args = { n, v, m_inv, (*q)[0], (*q)[1], (*q)[2], (*q)[3] };
    return node_call_fn(self.ltc_fn, &args);
}

fn Node ltc_edge(Node v1, Node v2) @local {
    Node x = nodes::dot(v1, v2);
    Node y = nodes::abs(x);
    Node a = (y * 0.0145206f + 0.4965155f) * y + 0.8543985f;
    Node b = (y + 4.1616724f) * y + 3.4175940f;
    Node k = a / b;
    Node far = nodes::inverse_sqrt(nodes::max((x * x).one_minus(), 1e-7f)) * 0.5f - k;
    return nodes::cross(v1, v2) * nodes::select(x.greater_than(nodes::num(0)), k, far);
}

// The light of `radiance` from direction `l` (toward it) on the surface: its diffuse light and
// its highlight (shader_core.glsl's direct(), three's physically based light units).
fn Node[2] Lit.direct(&self, Node l, Node radiance, Sized* size = null) {
    MaterialKey key = self.key;
    Node n = self.n;
    Node v = self.v;
    Node ndl = nodes::dot(n, l);
    Node nol;
    if (key.foliage) {
        // thin leaves and blades, lit from either side: on the side the light falls on, as any
        // surface, with a little of it wrapping round; from the other, about half of it
        Node through = nodes::select((nodes::dot(n, v) * ndl).less_than(nodes::num(0)), 0.5f, 1.0f);
        nol = through * (nodes::abs(ndl) * 0.9f + 0.1f);
    } else {
        nol = nodes::clamp(ndl, 0, 1);
    }
    // the toon material: the gradient's step in place of n.l, and no highlights
    if (key.type == (char)MaterialType.TOON) return { toon_gradient(key, ndl) * radiance * self.diffuse * shading::RECIPROCAL_PI, { } };
    // (a tube's cosine over the squared distance, integrated along it)
    Node irradiance = size && size.nol.id ? nodes::select(size.nol.greater_than_equal(nodes::num(0)), size.nol, nol) * radiance : nol * radiance;
    Node h = nodes::normalize(l + v);
    Node sized = size && size.on.id ? size.on : (Node){ };
    if (key.clearcoat) {
        // the coat's own highlight, off its own normal (f0 0.04: an IOR of 1.5); a light with a
        // size: the point of it the surface under the coat mirrors, as bright as the coat's
        // roughness keeps it
        Node cn = self.cc_n;
        Node a = self.cc_rough * self.cc_rough;
        Node cnl = nodes::clamp(nodes::dot(cn, l), 0, 1);
        Node cnv = nodes::clamp(nodes::dot(cn, v), 1e-4f, 1);
        Node cnorm = sized.id ? nodes::select(sized, light_norm(size.k, a), 1.0f) : nodes::num(1);
        Node cl = sized.id ? nodes::select(sized, size.l, l) : l;
        Node ch = nodes::normalize(cl + v);
        Node cnh = nodes::clamp(nodes::dot(cn, ch), 0, 1);
        Node fc = schlick(nodes::num(0.04f), nodes::num(1), nodes::clamp(nodes::dot(v, ch), 0, 1));
        Node ccl = nodes::clamp(nodes::dot(cn, cl), 0, 1);
        self.coat_d = cnl * radiance * fc * v_smith(cnv, ccl, a) * d_ggx(cnh, a) * cnorm;
    }
    if (key.sheen) {
        Node snv = nodes::clamp(nodes::dot(n, v), 0, 1);
        Node snl = nodes::clamp(nodes::dot(n, l), 0, 1);
        self.sheen_d = irradiance * self.sheen_color * d_charlie(self.sheen_rough, nodes::clamp(nodes::dot(n, h), 0, 1)) * v_neubelt(snv, snl);
        // what the fibers take is not left for the surface under them
        Node albedo = nodes::max(sheen_albedo(n, v, self.sheen_rough), sheen_albedo(n, l, self.sheen_rough));
        irradiance = irradiance * (max3(self.sheen_color) * albedo).one_minus();
    }
    Node result = irradiance * self.diffuse * shading::RECIPROCAL_PI;
    if (key.type == (char)MaterialType.LAMBERT) return { result, { } };
    Node noh = nodes::clamp(nodes::dot(n, h), 0, 1);
    Node voh = nodes::clamp(nodes::dot(v, h), 0, 1);
    if (key.type == (char)MaterialType.PHONG) {
        // (at least 1e-4 as three's: pow(0, 0) is undefined)
        Node shininess = nodes::max(self.shininess, 1e-4f);
        Node d = shading::RECIPROCAL_PI * (shininess * 0.5f + 1.0f) * nodes::pow(noh, shininess);
        Node f0 = self.phong_specular;
        Node fresnel = f0 + (nodes::vec3(1) - f0) * nodes::pow(voh.one_minus(), 5);
        return { result, irradiance * fresnel * 0.25f * d };
    }
    Node rough = nodes::max(self.rough, 0.0525f);
    Node a = rough * rough;
    Node snl = nol;
    Node sl = l;
    Node norm;
    if (sized.id) {
        // a light with a size: the highlight is the light's own shape, seen in the surface
        // (light_toward), as bright in all as the point light's
        Node sh = nodes::normalize(size.l + v);
        h = nodes::select(sized, sh, h);
        sl = nodes::select(sized, size.l, l);
        noh = nodes::select(sized, nodes::clamp(nodes::dot(n, sh), 0, 1), noh);
        voh = nodes::select(sized, nodes::clamp(nodes::dot(v, sh), 0, 1), voh);
        snl = nodes::select(sized, nodes::clamp(nodes::dot(n, size.l), 0, 1), nol);
        norm = nodes::select(sized, light_norm(size.k, a), 1.0f);
    }
    Node nov = nodes::clamp(nodes::abs(nodes::dot(n, v)), 1e-4f, 1);
    Node fresnel = self.f0 + (nodes::vec3(self.f90) - self.f0) * nodes::pow(voh.one_minus(), 5);
    if (key.iridescence) fresnel = nodes::select(self.irid.greater_than(nodes::num(0)), nodes::mix(fresnel, self.irid_fresnel, self.irid), fresnel);
    Node vis = v_smith(nov, snl, a);
    Node d = d_ggx(noh, a);
    if (key.anisotropy) {
        // rougher along aniso_t, by as much as it is anisotropic (three's alphaT)
        Node at = nodes::max(nodes::mix(a, 1, self.aniso * self.aniso), a);
        Node t = self.aniso_t;
        Node b = self.aniso_b;
        Node on = self.aniso.greater_than(nodes::num(0));
        vis = nodes::select(on, v_smith_aniso(at, a, nodes::dot(t, v), nodes::dot(b, v), nodes::dot(t, sl), nodes::dot(b, sl), nov, snl), vis);
        d = nodes::select(on, d_ggx_aniso(at, a, noh, nodes::dot(t, h), nodes::dot(b, h)), d);
    }
    Node highlight = irradiance * fresnel * vis * d;
    return { result, norm.id ? highlight * norm : highlight };
}

// The light the highlight of a light with a size keeps at roughness alpha `a`: a / a' along a
// tube (eq. 20) times (a / a')^2 for its ball (eq. 14), a' = saturate(a + half the length, or
// the radius, / (3 distance)) (eq. 10); `k` from light_toward.
fn Node light_norm(Node k, Node a) @local {
    Node kb = a / nodes::clamp(a + k.y(), 0, 1);
    return a / nodes::clamp(a + k.x(), 0, 1) * kb * kb;
}

// The light the scene's environment puts on a surface facing n, but for its sun (env_sun):
// env_sh's 9 spherical harmonics, in the order of texture.c3's sh_basis.
fn Node env_irradiance(Node n) @local {
    Node e = fs_in("env_sh", VEC3) + fs_in("env_sh", VEC3, k: 1) * n.y() + fs_in("env_sh", VEC3, k: 2) * n.z()
        + fs_in("env_sh", VEC3, k: 3) * n.x() + fs_in("env_sh", VEC3, k: 4) * (n.x() * n.y())
        + fs_in("env_sh", VEC3, k: 5) * (n.y() * n.z()) + fs_in("env_sh", VEC3, k: 6) * (3.0f * n.z() * n.z() - 1.0f)
        + fs_in("env_sh", VEC3, k: 7) * (n.x() * n.z()) + fs_in("env_sh", VEC3, k: 8) * (n.x() * n.x() - n.y() * n.y());
    return nodes::max(e, nodes::vec3(0));
}

fn Node env_sun_dir() @local => nodes::vec3(fs_in("env_sh", FLOAT, 3), fs_in("env_sh", FLOAT, 3, k: 1), fs_in("env_sh", FLOAT, 3, k: 2));

// The environment's sun on a surface facing n.
fn Node env_sun(Node n) @local {
    Node light = nodes::vec3(fs_in("env_sh", FLOAT, 3, k: 3), fs_in("env_sh", FLOAT, 3, k: 4), fs_in("env_sh", FLOAT, 3, k: 5));
    return light * nodes::max(nodes::dot(n, env_sun_dir()), 0);
}

// Where a rough surface's reflection mostly comes from: the mirror direction r, leaning towards
// the normal as the surface gets rougher (three's getIBLRadiance).
fn Node dominant_reflection(Node r, Node n, Node rough) @local {
    Node r2 = rough * rough;
    return nodes::normalize(nodes::mix(r, n, r2 * r2));
}

// How widely a rough surface's reflection is blurred (radians), as three's PMREM blurs it.
fn Node env_blur(Node rough) @local {
    Node low = 1.37f * rough * rough;
    Node b = nodes::select(rough.greater_than_equal(nodes::num(0.21f)), nodes::mix(0.06f, 0.125f, (rough - 0.21f) / 0.095f), low);
    b = nodes::select(rough.greater_than_equal(nodes::num(0.305f)), nodes::mix(0.125f, 0.215f, (rough - 0.305f) / 0.095f), b);
    b = nodes::select(rough.greater_than_equal(nodes::num(0.4f)), nodes::mix(0.215f, 0.526f, (rough - 0.4f) / 0.4f), b);
    return nodes::select(rough.greater_than_equal(nodes::num(0.8f)), nodes::mix(0.526f, 0.582f, (rough - 0.8f) / 0.2f), b);
}

// What the surface mirrors in direction d, blurred for `rough` (shader_core.glsl's
// env_reflection): the material's own map, box projected (the ray met with the room's walls
// first, or with what a map with distances saw in the room); or the scene's reflection probes
// (key.probes), mixed by the boxes `world` is in. `facing` is the surface's normal, `frag` the
// pixel's position in the target; `probe_fn` the probes' function (-1: not made yet).
fn Node env_sample(MaterialKey key, int* probe_fn, Node d, Node rough, Node world, Node frag, Node facing) @local {
    if (key.probes) return probe_reflection(probe_fn, d, rough, world, frag, facing);
    if (key.env_box) {
        EnvSource own = { .pos = fs_in("env_pos", VEC3), .lo = fs_in("env_box_min", VEC3), .hi = fs_in("env_box_max", VEC3),
                          .range = fs_in("env_params", FLOAT, 3) };
        d = own.project(d, rough, world, frag);
    }
    return env_read({ }, d, rough);
}

// A map a reflection is read from: where it was captured, its box, the range of its
// distances (0: none) and, for a probe, its layer of the probes' texture (none: the
// material's own map).
struct EnvSource @local {
    Node layer;
    Node pos;
    Node lo;
    Node hi;
    Node range;
}

// The direction from where the map was captured to where the ray from `world` in direction
// d meets the room (box projection).
fn Node EnvSource.project(&self, Node d, Node rough, Node world, Node frag) {
    Node inv = 1.0f / d;
    Node t_far = nodes::max((self.hi - world) * inv, (self.lo - world) * inv);
    Node t = nodes::max(nodes::min(nodes::min(t_far.x(), t_far.y()), t_far.z()), 0);
    Node o = world - self.pos;
    return nodes::normalize(o + d * self.march(o, d, t, rough, frag));
}

// The map in direction d at the level `rough` blurs it to.
fn Node env_read(Node layer, Node d, Node rough) @local {
    Node size = layer.id ? node_sample(SIZE, shaders::VIEW_TEX_PROBES, shaders::SMP_ENV_SMP, TEX_ARRAY, nodes::int_(0)) : shading::env_size();
    Node width = nodes::convert(size.x(), FLOAT);
    return env_texel(layer, shading::env_uv(d), shading::env_lod(env_blur(rough), width)).xyz() * fs_in("env_params", FLOAT, 1);
}

fn Node env_texel(Node layer, Node uv, Node lod) @local {
    if (!layer.id) return shading::env_level(uv, lod);
    return node_sample(LEVEL, shaders::VIEW_TEX_PROBES, shaders::SMP_ENV_SMP, TEX_ARRAY, nodes::vec3(uv, layer), lod);
}

// The scene's reflection probes in direction d: the probe whose box `world` is in (or just
// outside of, facing in: a window pane in the room's wall), cross-faded with its neighbour
// over the blend width outside the box, and with the probe for everywhere else (its box
// empty) beyond that.
fn Node probe_reflection(int* probe_fn, Node d, Node rough, Node world, Node frag, Node facing) @local {
    Node first = nodes::variable(nodes::int_(-1));
    Node second = nodes::variable(nodes::int_(-1));
    Node outer = nodes::variable(nodes::int_(-1));
    Node w1 = nodes::variable(nodes::num(0));
    Node w2 = nodes::variable(nodes::num(0));
    Node i = nodes::loop_index(INT);
    Node lo = fs_at("probe_min", VEC4, i);
    Node hi = fs_at("probe_max", VEC4, i);
    Node empty = nodes::dot(nodes::abs(hi.xyz() - lo.xyz()), nodes::vec3(1)).equal(nodes::num(0));
    Node below = world - lo.xyz();
    Node above = hi.xyz() - world;
    Node inside = nodes::min(nodes::min(below.x(), nodes::min(below.y(), below.z())), nodes::min(above.x(), nodes::min(above.y(), above.z())));
    Node w = nodes::clamp(inside / nodes::max(lo.w(), 1e-3f) + 1.0f, 0, 1);
    // the outer side of the room's own wall shows what is outside
    w = nodes::select(nodes::dot(facing, fs_at("probe_pos", VEC3, i) - world).less_than_equal(nodes::num(0)), 0.0f, w);
    Node rank = nodes::if_then(w.greater_than(w1),
                               nodes::block(second.assign(first), w2.assign(w1), first.assign(i), w1.assign(w)),
                               nodes::block(nodes::if_then(w.greater_than(w2), nodes::block(second.assign(i), w2.assign(w)))));
    Node body = nodes::block(nodes::if_then(empty, nodes::block(outer.assign(i), nodes::continue_()), nodes::block(rank)));
    Node pick = nodes::loop(i, nodes::int_(0), nodes::convert(fs_in("probe_info", FLOAT), INT), body);
    Node sum = w1 + w2;
    Node has_outer = outer.greater_than_equal(nodes::int_(0));
    // with a probe for everywhere else the boxes' shares stay as they are (it takes the rest);
    // without one they are made to add up to all of it
    Node rest = nodes::select(has_outer, nodes::max(sum.one_minus(), 0), 0.0f);
    Node norm = 1.0f / nodes::max(nodes::select(has_outer, nodes::max(sum, 1.0f), sum), 1e-4f);
    Node result = nodes::variable(nodes::vec3(0));
    Node[6] st = { first, second, outer, w1, w2, pick };
    Node[3] which = { first, second, outer };
    Node[3] share = { w1 * norm, w2 * norm, rest };
    Node[3] add;
    for (int k = 0; k < 3; k++) {
        Node seen = probe_sample(probe_fn, d, rough, world, frag, which[k], nodes::num(k == 2 ? 0.0f : 1.0f));
        Node on = nodes::logical_and(which[k].greater_than_equal(nodes::int_(0)), share[k].greater_than(nodes::num(0)));
        add[k] = nodes::if_then(on, nodes::block(result.assign(result + seen * (share[k] * fs_at("probe_max", FLOAT, which[k], 3)))));
    }
    return nodes::seq(nodes::block(st[0], st[1], st[2], st[3], st[4], st[5], result, add[0], add[1], add[2]), result);
}

// Probe `index`'s map in direction d, box projected where `boxed` is 1 (the function
// probe_sample of the program, made once: `fn_id` -1 makes it).
fn Node probe_sample(int* fn_id, Node d, Node rough, Node world, Node frag, Node index, Node boxed) @local {
    if (*fn_id < 0) {
        NodeType[6] types = { VEC3, FLOAT, VEC3, VEC2, INT, FLOAT };
        int f = node_func("probe_sample", VEC3, &types);
        Node pd = node_param(f, 0);
        Node pr = node_param(f, 1);
        Node pw = node_param(f, 2);
        Node pf = node_param(f, 3);
        Node pi = node_param(f, 4);
        Node pb = node_param(f, 5);
        Node pos = fs_at("probe_pos", VEC4, pi);
        EnvSource probe = { .layer = nodes::convert(pi, FLOAT), .pos = pos.xyz(), .lo = fs_at("probe_min", VEC3, pi),
                            .hi = fs_at("probe_max", VEC3, pi), .range = pos.w() };
        Node dir = nodes::variable(pd);
        Node box = nodes::if_then(pb.greater_than(nodes::num(0.5f)), nodes::block(dir.assign(probe.project(pd, pr, pw, pf))));
        node_func_body(f, nodes::seq(nodes::block(dir, box), env_read(probe.layer, dir, pr)));
        *fn_id = f;
    }
    Node[6] args = { d, rough, world, frag, index, boxed };
    return node_call_fn(*fn_id, &args);
}

// How far the ray from `o` (relative to where the map was captured) in direction d goes, `t`
// to the room's walls, before it goes behind something the map saw: a map with distances walks
// it through the room in 72 steps and stops there (not far behind: then it only passes at the
// back of it, out of the map's sight). Only for sharp reflections (rough < 0.3); t otherwise.
fn Node EnvSource.march(&self, Node o, Node d, Node t, Node rough, Node frag) {
    const int STEPS = 72;
    Node range = self.range;
    Node dt = t / (float)STEPS;
    // the distances come in steps of 1/255 of their range: nearer than that to a surface, a
    // ray is not yet behind it (or a wall seen at a glancing angle breaks up into rings, one
    // per step)
    Node slack = range * (1.5f / 255.0f);
    Node noise = shading_noise(frag + nodes::vec2(47.0f, 17.0f) * fs_in("ssr_params", FLOAT, 2));
    Node far = nodes::variable(t);
    Node prev = nodes::variable(nodes::num(0));
    Node at = nodes::variable(dt * (noise * 0.7f + 0.3f));
    Node lo = nodes::variable(prev);
    Node hi = nodes::variable(at);
    Node mid = (lo + hi) * 0.5f;
    Node k = nodes::loop_index(UINT);
    Node halve = nodes::loop(k, nodes::uint_(0), nodes::uint_(4),
                             nodes::block(nodes::if_then(self.behind(o + d * mid).greater_than(slack),
                                                         nodes::block(hi.assign(mid)), nodes::block(lo.assign(mid)))));
    Node behind = self.behind(o + d * at);
    Node hit = nodes::logical_and(behind.greater_than(slack), behind.less_than(dt + 0.12f + slack));
    Node i = nodes::loop_index(UINT);
    Node step = nodes::loop(i, nodes::uint_(0), nodes::uint_(STEPS),
                            nodes::block(nodes::if_then(hit, nodes::block(lo, hi, halve, far.assign(hi), nodes::break_())),
                                         prev.assign(at), at.assign(at + dt)));
    Node on = nodes::logical_and(range.greater_than(nodes::num(0)), rough.less_than(nodes::num(0.3f)));
    return nodes::seq(nodes::block(far, nodes::if_then(on, nodes::block(prev, at, step))), far);
}

// How far the point `q` (relative to where the map was captured) lies behind what the map saw
// in its direction (its alpha: the distance over the range).
fn Node EnvSource.behind(&self, Node q) {
    Node len = nodes::length(q);
    return len - env_texel(self.layer, shading::env_uv(q / len), nodes::num(0)).w() * self.range;
}

// Interleaved gradient noise (Jimenez's), as shader_core.glsl's ssr_noise.
fn Node shading_noise(Node p) @local
    => nodes::fract(nodes::fract(nodes::dot(p, nodes::vec2(0.06711056f, 0.00583715f))) * 52.9829189f);

// The frame of a normal map at `n`: the vertex tangents', or the cotangent frame of the map's
// uv, `duv1` and `duv2` its change from one pixel to the next (tangent_frame, three's
// getTangentFrame: t and b scaled together).
fn Node[2] tangent_frame(MaterialKey key, Node n, Node v_normal, Node world, Node tangent, Node duv1, Node duv2) @private {
    // a back face (its normal turned to the camera): the frame turned over too
    Node back = nodes::dot(n, v_normal).less_than(nodes::num(0));
    Node dpdx = nodes::dfdx(world);
    Node dpdy = nodes::dfdy(world);
    Node dp2perp = nodes::cross(dpdy, n);
    Node dp1perp = nodes::cross(n, dpdx);
    Node t = dp2perp * duv1.x() + dp1perp * duv2.x();
    Node b = dp2perp * duv1.y() + dp1perp * duv2.y();
    // (three's formula takes the screen's y up, as GL's dFdy; Metal's and D3D's go down, which
    // turns the frame around: its sign, the determinant's, keeps it the same on every backend)
    Node invmax = nodes::inverse_sqrt(nodes::max(nodes::max(nodes::dot(t, t), nodes::dot(b, b)), 1e-12f));
    invmax = nodes::select(nodes::dot(dpdx, dp2perp).less_than(nodes::num(0)), invmax.negate(), invmax);
    invmax = nodes::select(back, invmax.negate(), invmax);
    t = t * invmax;
    b = b * invmax;
    if (key.tangents) {
        // (a vertex without one, all zeros: the frame from the uv instead)
        Node vt = nodes::varying(tangent);
        Node vn = nodes::normalize(v_normal);
        Node tt = nodes::normalize(vt.xyz());
        Node tb = nodes::normalize(nodes::cross(vn, tt) * vt.w());
        tt = nodes::select(back, tt.negate(), tt);
        tb = nodes::select(back, tb.negate(), tb);
        Node has = nodes::dot(vt.xyz(), vt.xyz()).greater_than(nodes::num(0));
        t = nodes::select(has, tt, t);
        b = nodes::select(has, tb, b);
    }
    return { t, b };
}

// MeshToonMaterial's light: the gradient map (or two steps without one) by where the light
// falls, in place of n.l (three's getGradientIrradiance).
fn Node toon_gradient(MaterialKey key, Node ndl) @local {
    Node x = ndl * 0.5f + 0.5f;
    int slot = key.slot(MAP_GRADIENT) - 1;
    if (slot >= 0) {
        int view = shaders::VIEW_TEX_EXT0 + slot;
        Node uv = nodes::vec2(nodes::clamp(x, 0, 0.999f), 0);
        if (key.toon_nearest) {
            Node size = nodes::convert(node_sample(SIZE, view, shaders::SMP_TEX_SMP, TEX_2D, { }), VEC2);
            Node at = nodes::convert(nodes::fract(uv) * size, IVEC2);
            return node_sample(FETCH, view, shaders::SMP_TEX_SMP, TEX_2D, at, nodes::int_(0)).x();
        }
        return node_sample(LEVEL, view, shaders::SMP_TEX_SMP, TEX_2D, uv, nodes::num(0)).x();
    }
    Node fw = nodes::fwidth(x) * 0.5f;
    return nodes::mix(0.7f, 1.0f, nodes::smoothstep(nodes::num(0.7f) - fw, fw + 0.7f, x));
}

// three's window: the light fades out to nothing at `cutoff` (0: never), and the falloff by
// `decay` (shader_core.glsl's distance_window, distance_attenuation).
fn Node distance_window(Node d, Node cutoff) @local {
    Node r = d / cutoff;
    Node f = nodes::clamp((r * r * r * r).one_minus(), 0, 1);
    return nodes::select(cutoff.less_than_equal(nodes::num(0)), 1.0f, f * f);
}

fn Node distance_attenuation(Node d, Node cutoff, Node decay) @local {
    return distance_window(d, cutoff) / nodes::max(nodes::pow(d, decay), 0.01f);
}

// A light with a size, seen in reflections (light_toward): where it is (`on`), the direction
// toward the point of it its highlight comes from, and how much its shape widens one (x along a
// tube, y round a ball); and a tube's cosine over the squared distance (-1: not a tube).
struct Sized @local {
    Node on;
    Node l;
    Node k;
    Node nol;
}

// Karis, "Real Shading in Unreal Engine 4", 2013, the representative point: the point of the
// light nearest the ray `r` reflected off the surface. `lv` is from the surface to the light's
// centre, `seg` from its centre to one end (a tube), `radius` its thickness. A tube first takes
// the point of its segment nearest the ray (his eq. 19), then the ball of `radius` round that
// point the point of it nearest the ray (eq. 11).
fn Sized light_toward(Node r, Node lv, Node seg, Node radius, bool tube) @local {
    Node kx = nodes::num(0);
    if (tube) {
        Node l0 = lv - seg;
        Node ld = seg * 2.0f;
        Node rld = nodes::dot(r, ld);
        Node t = (nodes::dot(r, l0) * rld - nodes::dot(l0, ld)) / nodes::max(nodes::dot(ld, ld) - rld * rld, 1e-8f);
        Node has = nodes::dot(seg, seg).greater_than(nodes::num(0));
        lv = nodes::select(has, l0 + ld * nodes::clamp(t, 0, 1), lv);
        kx = nodes::select(has, nodes::length(seg) / (nodes::max(nodes::length(lv), 1e-4f) * 3.0f), 0.0f);
    }
    Node ball = radius.greater_than(nodes::num(0));
    Node ky = nodes::select(ball, radius / (nodes::max(nodes::length(lv), 1e-4f) * 3.0f), 0.0f);
    // (the light's centre to the ray: Karis' eq. 11 with its sign put right)
    Node to_ray = r * nodes::dot(lv, r) - lv;
    lv = nodes::select(ball, lv + to_ray * nodes::clamp(radius / nodes::max(nodes::length(to_ray), 1e-6f), 0, 1), lv);
    return { .l = nodes::normalize(lv), .k = nodes::vec2(kx, ky) };
}

fn Node d_ggx(Node noh, Node a) @local {
    Node a2 = a * a;
    Node d = noh * noh * (a2 - 1.0f) + 1.0f;
    return shading::RECIPROCAL_PI * a2 / (d * d);
}

fn Node v_smith(Node nov, Node nol, Node a) @local {
    Node a2 = a * a;
    Node gv = nol * nodes::sqrt(a2 + a2.one_minus() * nov * nov);
    Node gl = nov * nodes::sqrt(a2 + a2.one_minus() * nol * nol);
    return 0.5f / nodes::max(gv + gl, 1e-6f);
}

// The light cluster `p` is in (shader_core.glsl's cluster_of).
fn Node cluster_of(Node p) @local {
    Node grid = nodes::convert(fs_in("cluster_grid", VEC3), IVEC3);
    Node clip = fs_in("cluster_view_proj", MAT4) * nodes::vec4(p, 1);
    Node ndc = clip.xy() / nodes::max(nodes::abs(clip.w()), 1e-6f);
    Node gxy = grid.xy();
    Node cell = nodes::convert(nodes::floor((ndc * 0.5f + 0.5f) * nodes::convert(gxy, VEC2)), IVEC2);
    Node tile = nodes::clamp(cell, nodes::convert(nodes::int_(0), IVEC2), gxy - nodes::int_(1));
    Node depth = nodes::dot(p - fs_in("camera_pos", VEC3), fs_in("dir_cascade_fwd", VEC3));
    Node cd = fs_in("cluster_depth");
    Node slice = nodes::select(depth.less_than_equal(cd.x()), nodes::int_(0),
                               nodes::convert(nodes::floor(nodes::log(depth / cd.x()) * cd.y()), INT));
    slice = nodes::clamp(slice, nodes::int_(0), grid.z() - nodes::int_(1));
    return tile.x() + grid.x() * (tile.y() + grid.y() * slice);
}
