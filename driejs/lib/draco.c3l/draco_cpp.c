// A C interface to Google's Draco decoder (libdraco), for draco.c3. C++, compiled with -x c++.
#include <cstring>
#include <memory>
#include <string>

#include "draco/compression/decode.h"
#include "draco/mesh/mesh.h"
#include "draco/point_cloud/point_cloud.h"

namespace {

std::string last_error;

struct Decoded {
    std::unique_ptr<draco::PointCloud> pc;
    draco::Mesh* mesh;  // pc as a mesh, or null for a point cloud
};

template <typename T>
bool copy_values(const draco::PointAttribute* att, int components, uint32_t points, void* out) {
    T* dst = static_cast<T*>(out);
    for (uint32_t i = 0; i < points; i++) {
        if (!att->ConvertValue<T>(att->mapped_index(draco::PointIndex(i)), (int8_t)components, dst + (size_t)i * components)) return false;
    }
    return true;
}

}  // namespace

extern "C" {

const char* drc_error(void) { return last_error.c_str(); }

// The decoded mesh or point cloud in `data`, or null with drc_error() set.
void* drc_decode(const char* data, size_t len) {
    draco::DecoderBuffer buffer;
    buffer.Init(data, len);
    auto type = draco::Decoder::GetEncodedGeometryType(&buffer);
    if (!type.ok()) {
        last_error = type.status().error_msg_string();
        return nullptr;
    }
    draco::Decoder decoder;
    auto* d = new Decoded();
    if (type.value() == draco::TRIANGULAR_MESH) {
        auto mesh = decoder.DecodeMeshFromBuffer(&buffer);
        if (!mesh.ok()) {
            last_error = mesh.status().error_msg_string();
            delete d;
            return nullptr;
        }
        d->mesh = mesh.value().get();
        d->pc = std::move(mesh).value();
    } else if (type.value() == draco::POINT_CLOUD) {
        auto pc = decoder.DecodePointCloudFromBuffer(&buffer);
        if (!pc.ok()) {
            last_error = pc.status().error_msg_string();
            delete d;
            return nullptr;
        }
        d->mesh = nullptr;
        d->pc = std::move(pc).value();
    } else {
        last_error = "Unexpected geometry type.";
        delete d;
        return nullptr;
    }
    return d;
}

void drc_free(void* h) { delete static_cast<Decoded*>(h); }

bool drc_is_mesh(void* h) { return static_cast<Decoded*>(h)->mesh != nullptr; }

uint32_t drc_num_points(void* h) { return static_cast<Decoded*>(h)->pc->num_points(); }

uint32_t drc_num_faces(void* h) {
    Decoded* d = static_cast<Decoded*>(h);
    return d->mesh ? d->mesh->num_faces() : 0;
}

// The attribute with the given unique id, or -1.
int drc_attribute_by_unique_id(void* h, uint32_t id) {
    draco::PointCloud* pc = static_cast<Decoded*>(h)->pc.get();
    for (int i = 0; i < pc->num_attributes(); i++) {
        if (pc->attribute(i)->unique_id() == id) return i;
    }
    return -1;
}

// The first attribute of a semantic type (draco::GeometryAttribute::Type), or -1.
int drc_attribute_by_type(void* h, int type) {
    return static_cast<Decoded*>(h)->pc->GetNamedAttributeId((draco::GeometryAttribute::Type)type);
}

int drc_attribute_components(void* h, int att) {
    return static_cast<Decoded*>(h)->pc->attribute(att)->num_components();
}

// Writes the attribute's value for every point, num_points * components values, converted to
// `data_type` (a draco::DataType).
bool drc_attribute_data(void* h, int att, int data_type, void* out) {
    Decoded* d = static_cast<Decoded*>(h);
    const draco::PointAttribute* a = d->pc->attribute(att);
    int n = a->num_components();
    uint32_t points = d->pc->num_points();
    switch (data_type) {
        case draco::DT_INT8: return copy_values<int8_t>(a, n, points, out);
        case draco::DT_UINT8: return copy_values<uint8_t>(a, n, points, out);
        case draco::DT_INT16: return copy_values<int16_t>(a, n, points, out);
        case draco::DT_UINT16: return copy_values<uint16_t>(a, n, points, out);
        case draco::DT_INT32: return copy_values<int32_t>(a, n, points, out);
        case draco::DT_UINT32: return copy_values<uint32_t>(a, n, points, out);
        case draco::DT_FLOAT32: return copy_values<float>(a, n, points, out);
        default: return false;
    }
}

// Writes num_faces * 3 indices.
void drc_indices(void* h, uint32_t* out) {
    draco::Mesh* mesh = static_cast<Decoded*>(h)->mesh;
    if (!mesh) return;
    for (uint32_t f = 0; f < mesh->num_faces(); f++) {
        const draco::Mesh::Face& face = mesh->face(draco::FaceIndex(f));
        for (int j = 0; j < 3; j++) out[(size_t)f * 3 + j] = face[j].value();
    }
}

}  // extern "C"
