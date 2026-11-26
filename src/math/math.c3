// Core math and color utilities: 2D/3D/4D vectors, column-major 4x4 matrices, XYZ Euler
// angles, scalar helpers, and sRGB/linear color conversion. Right-handed Y-up coordinates,
// angles in radians unless a parameter is suffixed `_deg`, and `Mat4` indexed `m[col][row]`.
// Colors are 24-bit sRGB hex (`0xRRGGBB`) on the API surface and linear RGB in shaders and
// lighting; no heap allocations occur here.
module drie;
import std::math;

// Ratio of a circle's circumference to its diameter.
const float PI      = 3.14159265358979323846;
// Factor converting degrees to radians (`PI / 180`).
const float DEG2RAD = PI / 180.0f;
// Factor converting radians to degrees (`180 / PI`).
const float RAD2DEG = 180.0f / PI;

// Two-component 32-bit float vector (UVs, screen coordinates, 2D offsets).
alias Vector2 = float[<2>];
// Three-component 32-bit float vector (positions, directions, linear RGB colors).
alias Vector3 = float[<3>];
// Four-component 32-bit float vector (homogeneous coordinates, RGBA, packed uniforms).
alias Vec4 = float[<4>];
// Euler rotation angles in radians, applied in intrinsic XYZ order.
alias Euler = float[<3>];
// Column-major 4x4 matrix indexed `m[column][row]`.
typedef Mat4 = float[4][4];

// 24-bit sRGB color packed as `0xRRGGBB` (for example `0x44aa88`).
alias Color = uint;

// Assigns `(x, y, z)` to the vector in place.
fn void Vector3.set(&self, float x, float y, float z) => *self = { x, y, z };
// Copies `v` into the vector in place.
fn void Vector3.copy(&self, Vector3 v) => *self = v;
// Sets all three components to `s` in place.
fn void Vector3.set_scalar(&self, float s) => *self = { s, s, s };
// Returns the Euclidean distance from this vector to `v`.
fn float Vector3.distance_to(&self, Vector3 v) => (*self - v).length();
// Linearly interpolates this vector toward `v` by `alpha` in place.
fn void Vector3.lerp(&self, Vector3 v, float alpha) => *self += (v - *self) * alpha;

// Converts degrees to radians.
fn float deg_to_rad(float deg) => deg * DEG2RAD;
// Converts radians to degrees.
fn float rad_to_deg(float rad) => rad * RAD2DEG;

// Clamps `x` to the closed interval `[lo, hi]`.
fn float clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// ---- Colors

// Packs 0..1 sRGB channel values into a 24-bit hex `Color`.
fn Color rgb(float r, float g, float b) {
    uint ri = (uint)(clamp(r, 0, 1) * 255 + 0.5f);
    uint gi = (uint)(clamp(g, 0, 1) * 255 + 0.5f);
    uint bi = (uint)(clamp(b, 0, 1) * 255 + 0.5f);
    return ri << 16 | gi << 8 | bi;
}

fn float hue_to_rgb(float p, float q, float t) @local {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1.0f / 6) return p + (q - p) * 6 * t;
    if (t < 0.5f) return q;
    if (t < 2.0f / 3) return p + (q - p) * 6 * (2.0f / 3 - t);
    return p;
}

// Builds an sRGB hex `Color` from hue, saturation, and lightness in 0..1 (`h` wraps modulo 1).
fn Color hsl(float h, float s, float l) {
    h = h - math::floor(h);
    s = clamp(s, 0, 1);
    l = clamp(l, 0, 1);
    if (s == 0) return rgb(l, l, l);
    float q = l <= 0.5f ? l * (1 + s) : l + s - l * s;
    float p = 2 * l - q;
    return rgb(hue_to_rgb(p, q, h + 1.0f / 3), hue_to_rgb(p, q, h), hue_to_rgb(p, q, h - 1.0f / 3));
}

// Converts a single sRGB channel in 0..1 to linear light using the IEC 61966-2-1 curve.
fn float srgb_to_linear(float c) {
    return c <= 0.04045f ? c / 12.92f : math::pow((c + 0.055f) / 1.055f, 2.4f);
}

// Converts an sRGB hex `Color` to linear RGB floats in 0..1 for lighting and shaders.
fn Vector3 color_to_linear(Color c) {
    return {
        srgb_to_linear(((c >> 16) & 0xff) / 255.0f),
        srgb_to_linear(((c >> 8) & 0xff) / 255.0f),
        srgb_to_linear((c & 0xff) / 255.0f),
    };
}

// Converts linear RGB floats in 0..1 to an sRGB hex `Color`, clamping out-of-range channels.
fn Color linear_to_color(Vector3 c) {
    uint hex;
    for (int k = 0; k < 3; k++) {
        float x = clamp(c[k], 0, 1);
        x = x <= 0.0031308f ? x * 12.92f : 1.055f * math::pow(x, 1 / 2.4f) - 0.055f;
        hex = (hex << 8) | (uint)(x * 255 + 0.5f);
    }
    return (Color)hex;
}

// Unpacks an sRGB hex `Color` into 0..1 sRGB floats without gamma decoding.
fn Vector3 color_to_rgb(Color c) {
    return { ((c >> 16) & 0xff) / 255.0f, ((c >> 8) & 0xff) / 255.0f, (c & 0xff) / 255.0f };
}

// ---- Matrices

// Returns the 4x4 identity matrix.
fn Mat4 mat4_identity() {
    return { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
}

// Multiplies two column-major 4x4 matrices (`self * b`).
fn Mat4 Mat4.mul(self, Mat4 b) {
    Mat4 m;
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            m[col][row] = self[0][row] * b[col][0] + self[1][row] * b[col][1] + self[2][row] * b[col][2]
                + self[3][row] * b[col][3];
        }
    }
    return m;
}

// Transforms a 3D point by an affine matrix (`w = 1`, ignoring the bottom row).
fn Vector3 Mat4.transform_point(self, Vector3 v) {
    return {
        self[0][0] * v.x + self[1][0] * v.y + self[2][0] * v.z + self[3][0],
        self[0][1] * v.x + self[1][1] * v.y + self[2][1] * v.z + self[3][1],
        self[0][2] * v.x + self[1][2] * v.y + self[2][2] * v.z + self[3][2],
    };
}

// Multiplies a 4D column vector by this matrix (`self * v`).
fn Vec4 Mat4.mul_vec4(self, Vec4 v) {
    return {
        self[0][0] * v.x + self[1][0] * v.y + self[2][0] * v.z + self[3][0] * v.w,
        self[0][1] * v.x + self[1][1] * v.y + self[2][1] * v.z + self[3][1] * v.w,
        self[0][2] * v.x + self[1][2] * v.y + self[2][2] * v.z + self[3][2] * v.w,
        self[0][3] * v.x + self[1][3] * v.y + self[2][3] * v.z + self[3][3] * v.w,
    };
}

// Extracts the translation vector from column 3 of the matrix.
fn Vector3 Mat4.get_position(self) => { self[3][0], self[3][1], self[3][2] };

// Returns the transpose of this matrix.
fn Mat4 Mat4.transpose(self) {
    Mat4 m;
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) m[c][r] = self[r][c];
    }
    return m;
}

// Composes a transform matrix from translation, XYZ Euler rotation (radians), and scale.
fn Mat4 mat4_compose(Vector3 pos, Euler rot, Vector3 scale) {
    float a  = math::cos(rot.x);
    float b  = math::sin(rot.x);
    float c  = math::cos(rot.y);
    float d  = math::sin(rot.y);
    float e  = math::cos(rot.z);
    float f  = math::sin(rot.z);
    float ae = a * e;
    float af = a * f;
    float be = b * e;
    float bf = b * f;
    Mat4 m;
    m[0] = { c * e, af + be * d, bf - ae * d, 0 };
    m[1] = { -c * f, ae - bf * d, be + af * d, 0 };
    m[2] = { d, -b * c, a * c, 0 };
    m[3] = { pos.x, pos.y, pos.z, 1 };
    for (int i = 0; i < 3; i++) {
        m[0][i] *= scale.x;
        m[1][i] *= scale.y;
        m[2][i] *= scale.z;
    }
    return m;
}

// Extracts XYZ Euler angles (radians) from the upper-left 3x3 rotation of an unscaled matrix.
fn Euler euler_from_rotation(Mat4 m) {
    float m11 = m[0][0];
    float m12 = m[1][0];
    float m13 = m[2][0];
    float m22 = m[1][1];
    float m23 = m[2][1];
    float m32 = m[1][2];
    float m33 = m[2][2];
    Euler e;
    e.y = math::asin(clamp(m13, -1, 1));
    if (math::abs(m13) < 0.9999999f) {
        e.x = math::atan2(-m23, m33);
        e.z = math::atan2(-m12, m11);
    } else {
        e.x = math::atan2(m32, m22);
        e.z = 0;
    }
    return e;
}

// Builds a rotation matrix orienting `-Z` (when `is_camera`) or `+Z` (otherwise) from `eye`
// toward `target`, using `up` to fix roll. Nudges `z` when parallel to `up`.
fn Mat4 mat4_look_at(Vector3 eye, Vector3 target, Vector3 up, bool is_camera) {
    Vector3 z = is_camera ? eye - target : target - eye;
    if (z.length() < 1e-8f) z = { 0, 0, 1 };
    z = z.normalize();
    Vector3 x = up.cross(z);
    if (x.length() < 1e-8f) {
        // Nudge `z` away from `up` so the cross product stays non-degenerate.
        if (math::abs(up.z) == 1) {
            z.x += 0.0001f;
        } else {
            z.z += 0.0001f;
        }
        z = z.normalize();
        x = up.cross(z);
    }
    x = x.normalize();
    Vector3 y = z.cross(x);
    Mat4 m    = mat4_identity();
    m[0] = { x.x, x.y, x.z, 0 };
    m[1] = { y.x, y.y, y.z, 0 };
    m[2] = { z.x, z.y, z.z, 0 };
    return m;
}

// Builds a perspective projection matrix from vertical `fov_deg`, `aspect` (`width / height`),
// `near`/`far`, and `zoom`. `zero_to_one` selects `[0, 1]` (Metal/D3D) vs `[-1, 1]` (GL).
fn Mat4 mat4_perspective(float fov_deg, float aspect, float near, float far, float zoom, bool zero_to_one) {
    float f = 1.0f / math::tan(fov_deg * DEG2RAD * 0.5f) * zoom;
    Mat4 m;
    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][3] = -1;
    if (zero_to_one) {
        m[2][2] = far / (near - far);
        m[3][2] = near * far / (near - far);
    } else {
        m[2][2] = (far + near) / (near - far);
        m[3][2] = 2 * far * near / (near - far);
    }
    return m;
}

// Builds an orthographic projection matrix for `[left, right] x [bottom, top] x [near, far]`
// (`top` precedes `bottom`). `zero_to_one` selects `[0, 1]` vs `[-1, 1]` clip depth.
fn Mat4 mat4_orthographic(float left, float right, float top, float bottom, float near, float far, bool zero_to_one) {
    Mat4 m = mat4_identity();
    m[0][0] = 2 / (right - left);
    m[1][1] = 2 / (top - bottom);
    m[3][0] = -(right + left) / (right - left);
    m[3][1] = -(top + bottom) / (top - bottom);
    if (zero_to_one) {
        m[2][2] = -1 / (far - near);
        m[3][2] = -near / (far - near);
    } else {
        m[2][2] = -2 / (far - near);
        m[3][2] = -(far + near) / (far - near);
    }
    return m;
}

// Returns the inverse of this matrix, or `mat4_identity()` if the determinant is zero.
fn Mat4 Mat4.invert(self) {
    float[16] a, inv;
    for (int i = 0; i < 16; i++) a[i] = self[i / 4][i % 4];

    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14]
        + a[13] * a[6] * a[11]
        - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14]
        - a[12] * a[6] * a[11]
        + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13]
        + a[12] * a[5] * a[11]
        - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13]
        - a[12] * a[5] * a[10]
        + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14]
        - a[13] * a[2] * a[11]
        + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14]
        + a[12] * a[2] * a[11]
        - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13]
        - a[12] * a[1] * a[11]
        + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13]
        + a[12] * a[1] * a[10]
        - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14]
        + a[13] * a[2] * a[7]
        - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14]
        - a[12] * a[2] * a[7]
        + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13]
        + a[12] * a[1] * a[7]
        - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13]
        - a[12] * a[1] * a[6]
        + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10]
        - a[9] * a[2] * a[7]
        + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7]
        - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7]
        + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6]
        - a[8] * a[2] * a[5];

    float det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if (det == 0) return mat4_identity();
    det = 1.0f / det;
    Mat4 m;
    for (int i = 0; i < 16; i++) m[i / 4][i % 4] = inv[i] * det;
    return m;
}