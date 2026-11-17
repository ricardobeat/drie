// Google's Draco decoder (libdraco, from Homebrew), through draco_cpp.c.
module draco;

// draco::DataType
constdef DataType : int
{
    INT8 = 1,
    UINT8 = 2,
    INT16 = 3,
    UINT16 = 4,
    INT32 = 5,
    UINT32 = 6,
    FLOAT32 = 9,
}

// draco::GeometryAttribute::Type
constdef AttributeType : int
{
    POSITION = 0,
    NORMAL = 1,
    COLOR = 2,
    TEX_COORD = 3,
    GENERIC = 4,
}

alias Decoded = void*;

extern fn ZString error() @cname("drc_error");
extern fn Decoded decode(char* data, usz len) @cname("drc_decode");
extern fn void free(Decoded d) @cname("drc_free");
extern fn bool is_mesh(Decoded d) @cname("drc_is_mesh");
extern fn uint num_points(Decoded d) @cname("drc_num_points");
extern fn uint num_faces(Decoded d) @cname("drc_num_faces");
extern fn int attribute_by_unique_id(Decoded d, uint id) @cname("drc_attribute_by_unique_id");
extern fn int attribute_by_type(Decoded d, AttributeType type) @cname("drc_attribute_by_type");
extern fn int attribute_components(Decoded d, int att) @cname("drc_attribute_components");
extern fn bool attribute_data(Decoded d, int att, DataType data_type, void* out) @cname("drc_attribute_data");
extern fn void indices(Decoded d, uint* out) @cname("drc_indices");
