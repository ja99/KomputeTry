import pyshader as ps
from pyshader import *
import numpy as np
import kp
from pyshader.stdlib import exp, log
import plotly.express as px

Pixel = ps.Struct(
    x=ps.u8,
    y=ps.u8,
    color=ps.vec3
)

Texture = ps.Array(ps.vec3)


@ps.python2shader
def compute_shader(
        index=("input", "GlobalInvocationId", ps.ivec3),
        in_tex_size=("buffer", 0, ps.Array(ps.f32)),
        out_tex=("buffer", 1, ps.Array(ps.f32))
        # out_tex=("buffer", 1, ps.Array(ps.f32))
):
    r = f32(index.x)
    g = f32(index.y)
    b = 0.0

    uid = (index.x * i32(in_tex_size[0]) + index.y) * 3

    out_tex[uid + 0] = r
    out_tex[uid + 1] = g
    out_tex[uid + 2] = b


def test_logistic_regression():
    mgr = kp.Manager(0)
    workgroup = (*tex_size, 1)

    # First we create input and ouput tensors for shader
    tensor_in_a = mgr.tensor(np.array(tex_size, dtype=np.float32))

    # tensor_out_a = mgr.tensor(np.zeros((*tex_size,3), dtype=np.float32))
    tensor_out_a = mgr.tensor(np.zeros((*tex_size, 3), dtype=np.float32))

    params = [tensor_in_a, tensor_out_a]

    mgr.sequence().eval(kp.OpTensorSyncDevice(params))

    # Create a managed sequence
    sq = mgr.sequence()

    # Record operation to sync memory from local to GPU memory
    # sq.record(kp.OpTensorSyncDevice([tensor_push_a]))

    # Record operation to execute GPU shader against all our parameters
    sq.record(kp.OpAlgoDispatch(mgr.algorithm(params, compute_shader.to_spirv(), workgroup)))

    # Record operation to sync memory from GPU to local memory
    sq.record(kp.OpTensorSyncLocal([tensor_out_a]))

    sq.eval()

    out_data: np.ndarray = tensor_out_a.data()

    out_data = np.reshape(out_data, (*tex_size, 3))
    # out_data = np.reshape(out_data, tex_size)

    # px.imshow(out_data).show()
    px.imshow(out_data, range_color=(0, tex_size[0])).show()

    print(tensor_out_a.data().shape)
    print(tensor_out_a.data())


if __name__ == "__main__":
    tex_size = (2048, 2048)
    test_logistic_regression()
    print("All tests passed!")
