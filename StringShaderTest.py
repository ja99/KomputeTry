from pathlib import Path

import numpy as np
import kp

import os


def compile_source(source: Path):
    os.system(f"glslangValidator -V {source} -o {source}.spv")
    return open(f"{source}.spv", "rb").read()


def kompute(shader: Path):
    # 1. Create Kompute Manager with default settings (device 0, first queue and no extensions)
    mgr = kp.Manager()

    # 2. Create and initialise Kompute Tensors through manager

    # Define your spheres and query point
    spheres = np.array([
        [1.0, 2.0, 3.0, 1.0],  # Sphere with position (1,2,3) and radius 1
        [4.0, 5.0, 6.0, 2.0],  # Sphere with position (4,5,6) and radius 2
        [0.0, 0.0, 1.0, 0.5],
    ], dtype=np.float32)

    # Query point
    query_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Default tensor constructor simplifies creation of float values
    tensor_spheres = mgr.tensor(spheres)
    tensor_output = mgr.tensor(np.inf)  # For storing the result

    params = [tensor_spheres, tensor_output]


    # 3. Create algorithm based on shader (supports buffers & push/spec constants)
    workgroup = (spheres.shape[0], 1, 1)
    spec_consts = []
    push_consts = query_point

    # See documentation shader section for compile_source
    compiled_shader = compile_source(shader)

    algo = mgr.algorithm(params, compiled_shader, workgroup, spec_consts, push_consts)

    # 4. Run operation synchronously using sequence
    (mgr.sequence()
     .record(kp.OpTensorSyncDevice(params))
     .record(kp.OpAlgoDispatch(algo))  # Binds default push consts provided
     .record(kp.OpTensorSyncLocal(params))
     .eval()  # evaluates the two recorded ops
     )

    # # 5. Sync results from the GPU asynchronously
    # sq = mgr.sequence()
    # sq.eval_async(kp.OpTensorSyncLocal(params))
    #
    # # ... Do other work asynchronously whilst GPU finishes
    #
    # sq.eval_await()

    print(tensor_output.data())



if __name__ == "__main__":
    shader = Path("test_shader.comp")

    kompute(shader)
