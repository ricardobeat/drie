// Where a sound is heard from: an AudioListener (the ears) and a PositionalAudio (the source),
// named after three.js's. The engine does not play anything itself; a mixer asks the listener
// for a source's StereoPan and applies it to the samples it mixes:
//
//   AudioListener ears;
//   ears.set(camera.position, yaw);                       // or ears.set_from(camera)
//   PositionalAudio click = { .position = at, .ref_distance = 8 };
//   StereoPan p = ears.pan(click);
//   left  = p.left  * sample(now - p.delay_left);         // + a low-pass by p.shadow_left
//   right = p.right * sample(now - p.delay_right);
//   surround = p.behind * ...                             // see StereoPan.behind
module drie;
import std::math;

struct AudioListener {
    Vector3 position;
    Vector3 forward;
    Vector3 right;
}

// Ears at `position`, facing along `yaw` (radians around +y; 0 looks down -z).
fn void AudioListener.set(&self, Vector3 position, float yaw) {
    self.position = position;
    self.forward = { -math::sin(yaw), 0, -math::cos(yaw) };
    self.right = { math::cos(yaw), 0, -math::sin(yaw) };
}

// Ears on an object, usually the camera: where it is and the way it looks.
fn void AudioListener.set_from(&self, Object3D* obj) {
    Mat4 m = obj.matrix_world;
    self.position = m.get_position();
    self.forward = (m.transform_point({ 0, 0, -1 }) - self.position).normalize();
    self.right = (m.transform_point({ 1, 0, 0 }) - self.position).normalize();
}

// A sound somewhere in the world. Its level falls off like three's "inverse" distance model:
// full within ref_distance, ref / (ref + rolloff * (d - ref)) beyond, and no lower than
// what it is at max_distance (0 = no limit).
struct PositionalAudio {
    Vector3 position;
    float ref_distance;
    float rolloff_factor;
    float max_distance;
}

// How one source reaches the two ears.
struct StereoPan {
    // level at each ear, distance included
    float left;
    float right;
    // seconds the sound arrives later at each ear (one of them is 0)
    float delay_left;
    float delay_right;
    // 0..1: how much the head dulls each ear's highs (the far ear's)
    float shadow_left;
    float shadow_right;
    // 0..1: how far behind the listener it is. A mixer can dull it by this much, and feed
    // that share of it to the surround channel: +0.7 into left and -0.7 into right, which a
    // matrix (Pro Logic) decoder steers to the rear speakers.
    float behind;
}

// Half the width of a head over the speed of sound: the longest one ear waits for the other.
const float EAR_DELAY = 0.00066;

fn StereoPan AudioListener.pan(&self, PositionalAudio source) {
    Vector3 to      = source.position - self.position;
    float d         = to.length();
    float reference = math::max(source.ref_distance, 0.01f);
    float rolloff   = source.rolloff_factor > 0 ? source.rolloff_factor : 1.0f;
    float reach     = source.max_distance > 0 ? math::min(d, source.max_distance) : d;
    float gain      = reference / (reference + rolloff * (math::max(reach, reference) - reference));
    // on top of the listener: no direction
    float side      = 0;
    float front     = 1;
    if (d > 0.001f) {
        side = to.dot(self.right) / d;
        front = to.dot(self.forward) / d;
    }
    // equal power, never quite all in one ear
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