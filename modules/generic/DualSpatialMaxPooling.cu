#include "../utils.h"
#include "../common.h"

template <typename Dtype>
__global__ void DualMaxPoolForward(
		const int nthreads,
		const int batchSize,
		const int channels,
		const int iheight,
		const int iwidth,
		const int oheight,
		const int owidth,
		const int kH,
		const int kW,
		const int dH,
		const int dW,
		const int padH,
		const int padW,
		const Dtype* input_data,
		Dtype* output_data,
		Dtype* indices_data) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
		int pw = index % owidth;
		int ph = (index / owidth) % oheight;
		int c = (index / owidth / oheight) % channels;
		int n = index / owidth / oheight / channels;
		int hstart = ph * dH - padH;
		int wstart = pw * dW - padW;
		int hend = min(hstart + kH, iheight);
		int wend = min(wstart + kW, iwidth);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		Dtype maxval = -FLT_MAX;
		int maxidx = -1;
		input_data += (n * channels + c) * iheight * iwidth;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				if (input_data[h * iwidth + w] > maxval) {
					maxidx = h * iwidth + w;
					maxval = input_data[maxidx];
				}
			}
		}
		output_data[index] = maxval;
		indices_data[index] = maxidx + 1;
	}
}

static int unn_DualSpatialMaxPooling_updateOutput(lua_State *L) {
	THCState *state = getCutorchState(L);
	int kW = luaT_getfieldcheckint(L, 1, "kW");
	int kH = luaT_getfieldcheckint(L, 1, "kH");
	int dW = luaT_getfieldcheckint(L, 1, "dW");
	int dH = luaT_getfieldcheckint(L, 1, "dH");
	int padW = luaT_getfieldcheckint(L, 1, "padW");
	int padH = luaT_getfieldcheckint(L, 1, "padH");
	THCudaTensor *output = (THCudaTensor *) luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
	THCudaTensor *indices = (THCudaTensor *) luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");
	THCudaTensor *input = (THCudaTensor *) luaT_checkudata(L, 2, "torch.CudaTensor");

	THAssert(THCudaTensor_checkGPU(state, 3, input, output, indices));
	luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

	long batchSize, nInputPlane, iwidth, iheight, owidth, oheight;

	if (input->nDimension == 3) {
		batchSize = 1;
		nInputPlane = input->size[0];
		iheight = input->size[1];
		iwidth = input->size[2];
	} else {
		batchSize = input->size[0];
		nInputPlane = input->size[1];
		iheight = input->size[2];
		iwidth = input->size[3];
	}

	luaL_argcheck(L, iwidth >= kW - padW && iheight >= kH - padH, 2, "input image smaller than kernel size");
	luaL_argcheck(L, kW / 2 >= padW && kH / 2 >= padH, 2, "pad should be smaller than half of kernel size");

	owidth = floor(float(iwidth - kW + 2*padW) / float(dW)) + 1;
	oheight = floor(float(iheight - kH + 2*padH) / float(dH)) + 1;

	/* get contiguous input */
	input = THCudaTensor_newContiguous(state, input);
	/* resize */
	if (input->nDimension == 3) {
		THCudaTensor_resize3d(state, output, nInputPlane, oheight, owidth);
	} else {
		THCudaTensor_resize4d(state, output, batchSize, nInputPlane, oheight, owidth);
	}
	THCudaTensor_resizeAs(state, indices, output);

	int count = THCudaTensor_nElement(state, output);
	DualMaxPoolForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> (
		count, batchSize, nInputPlane, iheight, iwidth, oheight, owidth,
		kH, kW, dH, dW, padH, padW,
		THCudaTensor_data(state, input),
		THCudaTensor_data(state, output),
		THCudaTensor_data(state, indices)
	);

	// clean
	THCudaTensor_free(state, input);

	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in DualSpatialMaxPooling.updateOutput: %s\n", cudaGetErrorString(err));
		THError("aborting");
	}
	return 1;
}

template <typename Dtype>
__global__ void DualMaxPoolBackward(
	const int nthreads,
	const int batchSize,
	const int channels,
	const int iheight,
	const int iwidth,
	const int oheight,
	const int owidth,
	const int kH,
	const int kW,
	const int dH,
	const int dW,
	const int padH,
	const int padW,
	const Dtype* gradOutput_data,
	const Dtype* indices_data,
	Dtype* gradInput_data) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
		//int w = index % owidth;
		//int h = (index / owidth) % oheight;
		int c = (index / owidth / oheight) % channels;
		int n = index / owidth / oheight / channels;
		int maxPos = indices_data[index] - 1 + (n*channels + c)*iheight*iwidth;
		gradInput_data[maxPos] += gradOutput_data[index];
	}
}

static int unn_DualSpatialMaxPooling_updateGradInput(lua_State *L) {
	THCState *state = getCutorchState(L);
	int kW = luaT_getfieldcheckint(L, 1, "kW");
	int kH = luaT_getfieldcheckint(L, 1, "kH");
	int dW = luaT_getfieldcheckint(L, 1, "dW");
	int dH = luaT_getfieldcheckint(L, 1, "dH");
	int padW = luaT_getfieldcheckint(L, 1, "padW");
	int padH = luaT_getfieldcheckint(L, 1, "padH");
	THCudaTensor *gradInput = (THCudaTensor *) luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
	THCudaTensor *indices = (THCudaTensor *) luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");
	THCudaTensor *input = (THCudaTensor *) luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *gradOutput = (THCudaTensor *) luaT_checkudata(L, 3, "torch.CudaTensor");

	THAssert(THCudaTensor_checkGPU(state, 4, input, gradOutput, indices, gradInput));

	/* get contiguous input and gradOutput */
	input = THCudaTensor_newContiguous(state, input);
	gradOutput = THCudaTensor_newContiguous(state, gradOutput);
	/* resize gradInput */
	THCudaTensor_resizeAs(state, gradInput, input);
	/* zero gradInput */
	THCudaTensor_zero(state, gradInput);

	long batchSize, nInputPlane, iwidth, iheight, owidth, oheight;

	if (input->nDimension == 3) {
		batchSize = 1;
		nInputPlane = input->size[0];
		iheight = input->size[1];
		iwidth = input->size[2];
	} else {
		batchSize = input->size[0];
		nInputPlane = input->size[1];
		iheight = input->size[2];
		iwidth = input->size[3];
	}

	owidth = floor(float(iwidth - kW + 2*padW) / float(dW)) + 1;
	oheight = floor(float(iheight - kH + 2*padH) / float(dH)) + 1;

	int count = THCudaTensor_nElement(state, indices);
	DualMaxPoolBackward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> (
		count, batchSize, nInputPlane, iheight, iwidth, oheight, owidth,
		kH, kW, dH, dW, padH, padW,
		THCudaTensor_data(state, gradOutput),
		THCudaTensor_data(state, indices),
		THCudaTensor_data(state, gradInput)
	);

	// clean
	THCudaTensor_free(state, input);
	THCudaTensor_free(state, gradOutput);

	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in DualSpatialMaxPooling.updateGradInput: %s\n", cudaGetErrorString(err));
		THError("aborting");
	}
	return 1;
}

static const struct luaL_Reg unn_DualSpatialMaxPooling__[] = {
	{ "DualSpatialMaxPooling_updateOutput", unn_DualSpatialMaxPooling_updateOutput },
	{ "DualSpatialMaxPooling_updateGradInput", unn_DualSpatialMaxPooling_updateGradInput },
	{ NULL, NULL }
};

void unn_DualSpatialMaxPooling_init(lua_State *L) {
	luaT_pushmetatable(L, "torch.CudaTensor");
	luaT_registeratname(L, unn_DualSpatialMaxPooling__, "unn");
	lua_pop(L, 1);
}
