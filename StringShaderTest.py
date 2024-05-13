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

    # Default tensor constructor simplifies creation of float values
    tensor_in_a = mgr.tensor([2, 2, 2])
    tensor_in_b = mgr.tensor([1, 2, 3])
    # Explicit type constructor supports uint32, int32, double, float and bool
    tensor_out_a = mgr.tensor_t(np.array([0, 0, 0], dtype=np.uint32))
    tensor_out_b = mgr.tensor_t(np.array([0, 0, 0], dtype=np.uint32))

    params = [tensor_in_a, tensor_in_b, tensor_out_a, tensor_out_b]

    # 3. Create algorithm based on shader (supports buffers & push/spec constants)
    workgroup = (3, 1, 1)
    spec_consts = [2]
    push_consts_a = [2]
    push_consts_b = [3]

    # See documentation shader section for compile_source
    compiled_shader = compile_source(shader)

    algo = mgr.algorithm(params, compiled_shader, workgroup, spec_consts, push_consts_a)

    # 4. Run operation synchronously using sequence
    (mgr.sequence()
     .record(kp.OpTensorSyncDevice(params))
     .record(kp.OpAlgoDispatch(algo))  # Binds default push consts provided
     .eval()  # evaluates the two recorded ops
     .record(kp.OpAlgoDispatch(algo, push_consts_b))  # Overrides push consts
     .eval() # evaluates the two recorded ops
     )

    # 5. Sync results from the GPU asynchronously
    sq = mgr.sequence()
    sq.eval_async(kp.OpTensorSyncLocal(params))

    # ... Do other work asynchronously whilst GPU finishes

    sq.eval_await()

    # Prints the first output which is: { 4, 8, 12 }
    print(tensor_out_a.data())
    # Prints the first output which is: { 10, 10, 10 }
    print(tensor_out_b.data())


if __name__ == "__main__":
    shader = Path("test_shader.comp")

    kompute(shader)
