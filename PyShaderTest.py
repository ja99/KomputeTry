import pyshader as ps
from pyshader import *
import numpy as np
import kp
from pyshader.stdlib import exp, log


@ps.python2shader
def compute_shader(
        index=("input", "GlobalInvocationId", ps.ivec3),
        in_a: np.ndarray = ("buffer", 0, ps.Array(ps.f32)),
        in_b: np.ndarray=("buffer", 1, ps.Array(ps.f32)),
        out_a: np.ndarray=("buffer", 2, ps.Array(ps.f32)),
        out_b: np.ndarray=("buffer", 3, ps.Array(ps.f32)),
):
    i = index.x

    out_a[i] = in_a[i] * 5.0
    out_b[i] = in_b[i]


def test_logistic_regression():
    mgr = kp.Manager(0)

    # First we create input and ouput tensors for shader
    tensor_in_a = mgr.tensor(np.array([0, 1, 2, 3, 4], dtype=np.float32))
    tensor_in_b = mgr.tensor(np.array([4, 3, 2, 1, 0], dtype=np.float32))
    tensor_out_a = mgr.tensor(np.array([0, 0, 0, 0, 0], dtype=np.float32))
    tensor_out_b = mgr.tensor(np.array([0, 0, 0, 0, 0], dtype=np.float32))

    tensor_push_a = mgr.tensor(np.array([10, 11, 12, 13, 14], dtype=np.float32))

    # We store them in an array for easier interaction
    params = [tensor_in_a, tensor_in_b, tensor_out_a, tensor_out_b]

    mgr.sequence().eval(kp.OpTensorSyncDevice(params))

    # Create a managed sequence
    sq = mgr.sequence()

    # Record operation to sync memory from local to GPU memory
    sq.record(kp.OpTensorSyncDevice([tensor_push_a]))

    # Record operation to execute GPU shader against all our parameters
    sq.record(kp.OpAlgoDispatch(mgr.algorithm(params, compute_shader.to_spirv())))

    # Record operation to sync memory from GPU to local memory
    sq.record(kp.OpTensorSyncLocal([tensor_out_a, tensor_out_b]))

    ITERATIONS = 1

    # Perform machine learning training and inference across all input X and Y
    for i_iter in range(ITERATIONS):
        # Execute an iteration of the algorithm
        sq.eval()

        print(tensor_out_a.data())
        print(tensor_out_b.data())


if __name__ == "__main__":
    test_logistic_regression()
    print("All tests passed!")
