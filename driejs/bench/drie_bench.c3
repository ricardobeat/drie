// The transform cases of three_bench.js in drie, natively: the work drie-js does in C today.
//
//   ./bench/run.sh drie
module drie_bench;
import drie, std::io, std::time, std::math;

fn double time_frames(Scene* scene, Mesh*[] meshes, bool moving)
{
    int frames;
    time::Clock start = time::clock::now();
    double elapsed;
    while (frames < 3 || elapsed < 1000)
    {
        if (moving) foreach (i, m : meshes) m.rotation.y = (float)frames * 0.01f + (float)i;
        scene.update_matrix_world();
        frames++;
        elapsed = (double)start.to_now() / 1e6;
    }
    return elapsed / frames;
}

fn void main()
{
    foreach (n : (int[]){ 1000, 5000, 20000 })
    {
        Scene* scene = drie::scene();
        BufferGeometry* geometry = drie::box_geometry(1, 1, 1);
        Material* material = (Material*)drie::mesh_standard_material();
        Mesh*[] meshes = mem::new_array(Mesh*, n);
        Group*[] groups = mem::new_array(Group*, (sz)(n + 9) / 10);
        int side = (int)math::ceil(math::sqrt(n / 10.0));
        for (int g = 0; g * 10 < n; g++)
        {
            Group* group = drie::group();
            group.position = { (float)(g % side) * 4 - (float)side * 2, 0, (float)(g / side) * 4 - (float)side * 2 };
            scene.add(group);
            groups[g] = group;
            for (int i = 0; i < 10 && g * 10 + i < n; i++)
            {
                Mesh* m = drie::mesh(geometry, material);
                m.position = { (float)(i % 3) - 1, (float)(i / 3) * 0.5f, 0 };
                m.scale = { 0.3f, 0.3f, 0.3f };
                group.add(m);
                meshes[g * 10 + i] = m;
            }
        }
        scene.update_matrix_world();
        io::printfn("%-34s %6d  %9.3f ms", "transforms: all moving", n, time_frames(scene, meshes, true));
        io::printfn("%-34s %6d  %9.3f ms", "transforms: nothing moving", n, time_frames(scene, meshes, false));
        foreach (g : groups) g.matrix_world_auto_update = false;
        io::printfn("%-34s %6d  %9.3f ms", "transforms: frozen (no autoupdate)", n, time_frames(scene, meshes, false));
    }
}
