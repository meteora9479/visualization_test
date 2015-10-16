#include "../utils.h"
#include "../common.h"

template <typename Dtype>
__global__ void DualMaxUnpoolForward(
	const int nthreads,
	const int batchSize,
	const int channels,
	const int iheight,
	const int iwidth,
	const int oheight,
	const int owidth,
	const Dtype* indices_data,
	const Dtype* input_data,
	Dtype* output_data) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
		//int w = index % iwidth;
		//int h = (index / iwidth) % iheight;
		int c = (index / iwidth / iheight) % channels;
		int n = index / iwidth / iheight / channels;
		int maxPos = indices_data[index] - 1 + (n*channels + c)*oheight*owidth;
		output_data[maxPos] += input_data[index];
	}
}

static int unn_DualSpatialMaxUnpooling_updateOutput(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaTensor *indices = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

	THAssert(THCudaTensor_checkGPU(state, 3, indices, input, output));

	luaL_argcheck(L, input->nDimension == indices->nDimension, 2, "dimension of input must equal to dimension of indices");
	luaL_argcheck(L, output->nDimension == indices->nDimension, 3, "dimension of output must equal to dimension of indices");
	if (input->nDimension == 3) {
		luaL_argcheck(L, input->size[0] == indices->size[0], 2, "slices of input must equal to slices of indices");
		luaL_argcheck(L, output->size[0] == indices->size[0], 3, "slices of output must equal to slices of indices");
	} else {
		luaL_argcheck(L, input->size[0] == indices->size[0], 2, "batch-size of input must equal to batch-size of indices");
		luaL_argcheck(L, input->size[1] == indices->size[1], 2, "slices of input must equal to slices of indices");
		luaL_argcheck(L, output->size[0] == indices->size[0], 3, "batch-size of output must equal to batch-size of indices");
		luaL_argcheck(L, output->size[1] == indices->size[1], 3, "slices of output must equal to slices of indices");
	}

	long batchSize, nslices, iwidth, iheight, owidth, oheight;
	if (input->nDimension == 3) {
		batchSize = 1;
		nslices = input->size[0];
		iheight = input->size[1];
		iwidth = input->size[2];
		oheight = output->size[1];
		owidth = output->size[2];
	} else {
		batchSize = input->size[0];
		nslices = input->size[1];
		iheight = input->size[2];
		iwidth = input->size[3];
		oheight = output->size[2];
		owidth = output->size[3];
	}

	/* get contiguous input */
	input = THCudaTensor_newContiguous(state, input);
	/* zero output */
	THCudaTensor_zero(state, output);

	int count = THCudaTensor_nElement(state, indices);
	DualMaxUnpoolForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> (
		count, batchSize, nslices, iheight, iwidth, oheight, owidth,
		THCudaTensor_data(state, indices),
		THCudaTensor_data(state, input),
		THCudaTensor_data(state, output)
	);

	// clean
	THCudaTensor_free(state, input);

	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in DualSpatialMaxUnpooling.updateOutput: %s\n", cudaGetErrorString(err));
		THError("aborting");
	}
	return 1;
}

template <typename Dtype>
__global__ void DualMaxUnpoolBackward(
	const int nthreads,
	const int num,
	const int channels,
	const int iheight,
	const int iwidth,
	const int oheight,
	const int owidth,
	const Dtype* indices_data,
	const Dtype* gradOutput_data,
	Dtype* gradInput_data) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
		//int w = index % iwidth;
		//int h = (index / iwidth) % iheight;
		int c = (index / iwidth / iheight) % channels;
		int n = index / iwidth / iheight / channels;
		int maxPos = indices_data[index] - 1 + (n*channels + c)*oheight*owidth;
		gradInput_data[index] += gradOutput_data[maxPos];
	}
}

static int unn_DualSpatialMaxUnpooling_updateGradInput(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaTensor *indices = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *gradInput = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");

	THAssert(THCudaTensor_checkGPU(state, 4, indices, input, gradOutput, gradInput));

	/* get contiguous input and gradOutput */
	input = THCudaTensor_newContiguous(state, input);
	gradOutput = THCudaTensor_newContiguous(state, gradOutput);
	/* resize */
	THCudaTensor_resizeAs(state, gradInput, input);
	/* zero gradInput */
	THCudaTensor_zero(state, gradInput);

	long batchSize, nslices, iwidth, iheight, owidth, oheight;

	if (input->nDimension == 3) {
		batchSize = 1;
		nslices = input->size[0];
		iheight = input->size[1];
		iwidth = input->size[2];
		oheight = gradOutput->size[1];
		owidth = gradOutput->size[2];
	} else {
		batchSize = input->size[0];
		nslices = input->size[1];
		iheight = input->size[2];
		iwidth = input->size[3];
		oheight = gradOutput->size[2];
		owidth = gradOutput->size[3];
	}

	int count = THCudaTensor_nElement(state, indices);
	DualMaxUnpoolBackward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> (
		count, batchSize, nslices, iheight, iwidth, oheight, owidth,
		THCudaTensor_data(state, indices),
		THCudaTensor_data(state, gradOutput),
		THCudaTensor_data(state, gradInput)
	);

	// clean
	//THCudaTensor_free(state, input);
	THCudaTensor_free(state, gradOutput);

	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in DualSpatialMaxUnpooling.updateGradInput: %s\n", cudaGetErrorString(err));
		THError("aborting");
	}
	return 1;
}

static const struct luaL_Reg unn_DualSpatialMaxUnpooling__ [] = {
	{"DualSpatialMaxUnpooling_updateOutput", unn_DualSpatialMaxUnpooling_updateOutput},
	{"DualSpatialMaxUnpooling_updateGradInput", unn_DualSpatialMaxUnpooling_updateGradInput},
	{NULL, NULL}
};

void unn_DualSpatialMaxUnpooling_init(lua_State *L) {
	luaT_pushmetatable(L, "torch.CudaTensor");
	luaT_registeratname(L, unn_DualSpatialMaxUnpooling__, "unn");
	lua_pop(L,1);
}
