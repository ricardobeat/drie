// Spatial audio attenuation and stereo panning: `AudioListener` (listener pose) and
// `PositionalAudio` (point source) compute `StereoPan` parameters for an external audio mixer;
// drie does not output audio itself. Coordinates are world-space meters (right-handed, Y-up,
// yaw in radians around +Y with 0 facing -Z), and all structs are stack-allocated values.
// Connects to the scene graph through `AudioListener.set_from(Object3D*)`.
module drie;
import std::math;

// Listener position and orientation used to pan `PositionalAudio` sources.
// 
//   AudioListener ears;
//   ears.set(camera.position, yaw);                       // or ears.set_from(camera)
//   PositionalAudio click = { .position = at, .ref_distance = 8 };
//   StereoPan p = ears.pan(click);
//   left  = p.left  * sample(now - p.delay_left);         // + low-pass by p.shadow_left
//   right = p.right * sample(now - p.delay_right);
//   surround = p.behind * ...                             // see StereoPan.behind
struct AudioListener {
    // Listener location in world-space meters.
    Vector3 position;
    // Unit forward direction the listener faces.
    Vector3 forward;
    // Unit right direction (listener's right ear).
    Vector3 right;
}

// Places the listener at `position`, facing along `yaw` (radians around +Y; 0 looks down -Z).
fn void AudioListener.set(&self, Vector3 position, float yaw) {
    self.position = position;
    self.forward = { -math::sin(yaw), 0, -math::cos(yaw) };
    self.right = { math::cos(yaw), 0, -math::sin(yaw) };
}

// Copies listener position and `-Z`/`+X` basis from `obj.matrix_world` (typically a `Camera`).
// Reads `matrix_world` as stored; update world matrices before calling.
fn void AudioListener.set_from(&self, Object3D* obj) {
    Mat4 m = obj.matrix_world;
    self.position = m.get_position();
    self.forward = (m.transform_point({ 0, 0, -1 }) - self.position).normalize();
    self.right = (m.transform_point({ 1, 0, 0 }) - self.position).normalize();
}

// World-space sound source using the inverse distance model: full gain inside `ref_distance`,
// `ref / (ref + rolloff * (d - ref))` beyond, floored at the gain at `max_distance` (0 = none).
struct PositionalAudio {
    // Source location in world-space meters.
    Vector3 position;
    // Distance within which gain stays at 1 (clamped to at least 0.01 m).
    float ref_distance;
    // Attenuation rate beyond `ref_distance` (0 or negative defaults to 1).
    float rolloff_factor;
    // Distance where attenuation stops increasing (0 = no cap).
    float max_distance;
}

// Stereo gain, interaural time delay, head shadow, and rear weight for one sound source.
struct StereoPan {
    // Final gain for the left ear, including distance attenuation.
    float left;
    // Final gain for the right ear, including distance attenuation.
    float right;
    // Arrival delay in seconds at the left ear (0 when the source is to the left).
    float delay_left;
    // Arrival delay in seconds at the right ear (0 when the source is to the right).
    float delay_right;
    // Head-shadow factor in 0..1 for low-passing the left ear (higher when source is right).
    float shadow_left;
    // Head-shadow factor in 0..1 for low-passing the right ear (higher when source is left).
    float shadow_right;
    // Rear weight in 0..1: 0 in front of the listener, 1 straight behind. Mixers can muffle
    // highs or steer this share to a matrix surround channel (+0.7 left, -0.7 right).
    float behind;
}

// Maximum interaural delay in seconds (head width divided by speed of sound).
const float EAR_DELAY = 0.00066;

// Computes stereo gains, ear delays, head shadow, and rear weight for `source` from this pose.
fn StereoPan AudioListener.pan(&self, PositionalAudio source) {
    Vector3 to      = source.position - self.position;
    float d         = to.length();
    float reference = math::max(source.ref_distance, 0.01f);
    float rolloff   = source.rolloff_factor > 0 ? source.rolloff_factor : 1.0f;
    float reach     = source.max_distance > 0 ? math::min(d, source.max_distance) : d;
    float gain      = reference / (reference + rolloff * (math::max(reach, reference) - reference));
    // Source at the listener's position has no well-defined direction.
    float side      = 0;
    float front     = 1;
    if (d > 0.001f) {
        side = to.dot(self.right) / d;
        front = to.dot(self.forward) / d;
    }
    // Equal-power pan, keeping a slight bleed into the far ear.
    float angle = (side * 0.8f + 1) * PI / 4;
    return {
        .left = gain * math::cos(angle) * 1.4142f,
        .right = gain * math::sin(angle) * 1.4142f,
        .delay_left = math::max(side, 0.0f) * EAR_DELAY,
        .delay_right = math::max(-side, 0.0f) * EAR_DELAY,
        .shadow_left = math::max(side, 0.0f),
        .shadow_right = math::max(-side, 0.0f),
        .behind = math::max(-front, 0.0f),
    };
}