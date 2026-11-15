// Vector / matrix / color math for three.
module three;
import std::math;

const float PI = 3.14159265358979323846;
const float DEG2RAD = PI / 180.0f;
const float RAD2DEG = 180.0f / PI;

alias Vector2 = float[<2>];
alias Vector3 = float[<3>];
alias Vec4 = float[<4>];
// Euler angles in radians, applied in XYZ order (three's default).
alias Euler = float[<3>];
// Column-major 4x4 matrix, indexed m[column][row].
typedef Mat4 = float[4][4];

// Colors are sRGB hex values, like 0x44aa88.
alias Color = uint;

fn void Vector3.set(&self, float x, float y, float z) => *self = { x, y, z };
fn void Vector3.copy(&self, Vector3 v) => *self = v;
fn void Vector3.set_scalar(&self, float s) => *self = { s, s, s };
fn float Vector3.distance_to(&self, Vector3 v) => (*self - v).length();
fn void Vector3.lerp(&self, Vector3 v, float alpha) => *self += (v - *self) * alpha;

fn float deg_to_rad(float deg) => deg * DEG2RAD;
fn float rad_to_deg(float rad) => rad * RAD2DEG;

fn float clamp(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

// ----------------------------------------------------------------------------
// Colors
// ----------------------------------------------------------------------------

// Build a hex color from 0..1 sRGB components.
fn Color rgb(float r, float g, float b)
{
    uint ri = (uint)(clamp(r, 0, 1) * 255 + 0.5f);
    uint gi = (uint)(clamp(g, 0, 1) * 255 + 0.5f);
    uint bi = (uint)(clamp(b, 0, 1) * 255 + 0.5f);
    return ri << 16 | gi << 8 | bi;
}

fn float hue_to_rgb(float p, float q, float t) @local
{
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1.0f / 6) return p + (q - p) * 6 * t;
    if (t < 0.5f) return q;
    if (t < 2.0f / 3) return p + (q - p) * 6 * (2.0f / 3 - t);
    return p;
}

// Build a hex color from hue, saturation, lightness (all 0..1), like Color.setHSL.
fn Color hsl(float h, float s, float l)
{
    h = h - math::floor(h);
    s = clamp(s, 0, 1);
    l = clamp(l, 0, 1);
    if (s == 0) return rgb(l, l, l);
    float q = l <= 0.5f ? l * (1 + s) : l + s - l * s;
    float p = 2 * l - q;
    return rgb(hue_to_rgb(p, q, h + 1.0f / 3), hue_to_rgb(p, q, h), hue_to_rgb(p, q, h - 1.0f / 3));
}

fn float srgb_to_linear(float c)
{
    return c <= 0.04045f ? c / 12.92f : math::pow((c + 0.055f) / 1.055f, 2.4f);
}

// sRGB hex -> linear rgb (what the shader works in).
fn Vector3 color_to_linear(Color c)
{
    return {
        srgb_to_linear(((c >> 16) & 0xff) / 255.0f),
        srgb_to_linear(((c >> 8) & 0xff) / 255.0f),
        srgb_to_linear((c & 0xff) / 255.0f),
    };
}

fn Vector3 color_to_rgb(Color c)
{
    return { ((c >> 16) & 0xff) / 255.0f, ((c >> 8) & 0xff) / 255.0f, (c & 0xff) / 255.0f };
}

// ----------------------------------------------------------------------------
// Matrices
// ----------------------------------------------------------------------------

fn Mat4 mat4_identity()
{
    return { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
}

fn Mat4 Mat4.mul(self, Mat4 b)
{
    Mat4 m;
    for (int col = 0; col < 4; col++)
    {
        for (int row = 0; row < 4; row++)
        {
            m[col][row] = self[0][row] * b[col][0] + self[1][row] * b[col][1]
                        + self[2][row] * b[col][2] + self[3][row] * b[col][3];
        }
    }
    return m;
}

fn Vector3 Mat4.transform_point(self, Vector3 v)
{
    return {
        self[0][0] * v.x + self[1][0] * v.y + self[2][0] * v.z + self[3][0],
        self[0][1] * v.x + self[1][1] * v.y + self[2][1] * v.z + self[3][1],
        self[0][2] * v.x + self[1][2] * v.y + self[2][2] * v.z + self[3][2],
    };
}

fn Vector3 Mat4.get_position(self) => { self[3][0], self[3][1], self[3][2] };

fn Mat4 Mat4.transpose(self)
{
    Mat4 m;
    for (int c = 0; c < 4; c++)
    {
        for (int r = 0; r < 4; r++) m[c][r] = self[r][c];
    }
    return m;
}

// Same as three's Matrix4.compose with an XYZ Euler rotation.
fn Mat4 mat4_compose(Vector3 pos, Euler rot, Vector3 scale)
{
    float a = math::cos(rot.x);
    float b = math::sin(rot.x);
    float c = math::cos(rot.y);
    float d = math::sin(rot.y);
    float e = math::cos(rot.z);
    float f = math::sin(rot.z);
    float ae = a * e;
    float af = a * f;
    float be = b * e;
    float bf = b * f;
    Mat4 m;
    m[0] = { c * e,          af + be * d,    bf - ae * d,    0 };
    m[1] = { -c * f,         ae - bf * d,    be + af * d,    0 };
    m[2] = { d,              -b * c,         a * c,          0 };
    m[3] = { pos.x,          pos.y,          pos.z,          1 };
    for (int i = 0; i < 3; i++)
    {
        m[0][i] *= scale.x;
        m[1][i] *= scale.y;
        m[2][i] *= scale.z;
    }
    return m;
}

// Extract XYZ Euler angles from the (unscaled) rotation part of a matrix.
fn Euler euler_from_rotation(Mat4 m)
{
    float m11 = m[0][0];
    float m12 = m[1][0];
    float m13 = m[2][0];
    float m22 = m[1][1];
    float m23 = m[2][1];
    float m32 = m[1][2];
    float m33 = m[2][2];
    Euler e;
    e.y = math::asin(clamp(m13, -1, 1));
    if (math::abs(m13) < 0.9999999f)
    {
        e.x = math::atan2(-m23, m33);
        e.z = math::atan2(-m12, m11);
    }
    else
    {
        e.x = math::atan2(m32, m22);
        e.z = 0;
    }
    return e;
}

// Rotation matrix whose -Z axis (camera) or +Z axis (objects) points from eye to target.
fn Mat4 mat4_look_at(Vector3 eye, Vector3 target, Vector3 up, bool is_camera)
{
    Vector3 z = is_camera ? eye - target : target - eye;
    if (z.length() < 1e-8f) z = { 0, 0, 1 };
    z = z.normalize();
    Vector3 x = up.cross(z);
    if (x.length() < 1e-8f)
    {
        // up and z are parallel: nudge z
        if (math::abs(up.z) == 1) { z.x += 0.0001f; } else { z.z += 0.0001f; }
        z = z.normalize();
        x = up.cross(z);
    }
    x = x.normalize();
    Vector3 y = z.cross(x);
    Mat4 m = mat4_identity();
    m[0] = { x.x, x.y, x.z, 0 };
    m[1] = { y.x, y.y, y.z, 0 };
    m[2] = { z.x, z.y, z.z, 0 };
    return m;
}

// Perspective projection. zero_to_one selects Metal/D3D clip depth (0..1) vs GL (-1..1).
fn Mat4 mat4_perspective(float fov_deg, float aspect, float near, float far, float zoom, bool zero_to_one)
{
    float f = 1.0f / math::tan(fov_deg * DEG2RAD * 0.5f) * zoom;
    Mat4 m;
    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][3] = -1;
    if (zero_to_one)
    {
        m[2][2] = far / (near - far);
        m[3][2] = near * far / (near - far);
    }
    else
    {
        m[2][2] = (far + near) / (near - far);
        m[3][2] = 2 * far * near / (near - far);
    }
    return m;
}

fn Mat4 mat4_orthographic(float left, float right, float top, float bottom, float near, float far, bool zero_to_one)
{
    Mat4 m = mat4_identity();
    m[0][0] = 2 / (right - left);
    m[1][1] = 2 / (top - bottom);
    m[3][0] = -(right + left) / (right - left);
    m[3][1] = -(top + bottom) / (top - bottom);
    if (zero_to_one)
    {
        m[2][2] = -1 / (far - near);
        m[3][2] = -near / (far - near);
    }
    else
    {
        m[2][2] = -2 / (far - near);
        m[3][2] = -(far + near) / (far - near);
    }
    return m;
}

fn Mat4 Mat4.invert(self)
{
    float[16] a, inv;
    for (int i = 0; i < 16; i++) a[i] = self[i / 4][i % 4];

    inv[0] = a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    inv[4] = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    inv[8] = a[4]*a[9]*a[15] - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14] + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    inv[1] = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    inv[5] = a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    inv[9] = -a[0]*a[9]*a[15] + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    inv[13] = a[0]*a[9]*a[14] - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    inv[2] = a[1]*a[6]*a[15] - a[1]*a[7]*a[14] - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7] - a[13]*a[3]*a[6];
    inv[6] = -a[0]*a[6]*a[15] + a[0]*a[7]*a[14] + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7] + a[12]*a[3]*a[6];
    inv[10] = a[0]*a[5]*a[15] - a[0]*a[7]*a[13] - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7] - a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14] + a[0]*a[6]*a[13] + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6] + a[12]*a[2]*a[5];
    inv[3] = -a[1]*a[6]*a[11] + a[1]*a[7]*a[10] + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7] + a[9]*a[3]*a[6];
    inv[7] = a[0]*a[6]*a[11] - a[0]*a[7]*a[10] - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7] - a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11] + a[0]*a[7]*a[9] + a[4]*a[1]*a[11] - a[4]*a[3]*a[9] - a[8]*a[1]*a[7] + a[8]*a[3]*a[5];
    inv[15] = a[0]*a[5]*a[10] - a[0]*a[6]*a[9] - a[4]*a[1]*a[10] + a[4]*a[2]*a[9] + a[8]*a[1]*a[6] - a[8]*a[2]*a[5];

    float det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if (det == 0) return mat4_identity();
    det = 1.0f / det;
    Mat4 m;
    for (int i = 0; i < 16; i++) m[i / 4][i % 4] = inv[i] * det;
    return m;
}
