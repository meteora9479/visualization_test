#include "THCApply.cuh"
#include "../utils.h"
#include "../common.h"

struct ThresholdUpdateRecOutput {
	const float threshold_;
	const float val_;

	ThresholdUpdateRecOutput(float threshold, float val) :
			threshold_(threshold), val_(val) {
	}

	__device__ __forceinline__ void operator()(float* recOutput, float* input, float* recInput) const {
		*recOutput = ((*input > threshold_) && (*recInput > threshold_)) ? *recInput : val_;
	}
};

struct ThresholdUpdateRecOutputIP {
	const float threshold_;
	const float val_;

	ThresholdUpdateRecOutputIP(float threshold, float val) :
			threshold_(threshold), val_(val) {
	}

	__device__ __forceinline__ void operator()(float* recInput, float* input) const {
		*recInput = ((*input > threshold_) && (*recInput > threshold_)) ? *recInput : val_;
	}
};

static int dcnn_Threshold_updateRecOutput(lua_State *L) {
	THCState *state = getCutorchState(L);
	double val = luaT_getfieldchecknumber(L, 1, "val");
	double threshold = luaT_getfieldchecknumber(L, 1, "threshold");
	bool inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
	THCudaTensor *recOutput = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "recOutput", "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
	THCudaTensor *input = (THCudaTensor*) luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *recInput = (THCudaTensor*) luaT_checkudata(L, 3, "torch.CudaTensor");

	THAssert(THCudaTensor_checkGPU(state, 4, input, output, recOutput, recInput));

	if (inPlace) {
		THCudaTensor_pointwiseApply2(state, recInput, input, ThresholdUpdateRecOutputIP(threshold, val));
		THCudaTensor_set(state, recOutput, recInput);
	} else {
		THCudaTensor_resizeAs(state, recOutput, output);
		THCudaTensor_pointwiseApply3(state, recOutput, input, recInput, ThresholdUpdateRecOutput(threshold, val));
	}

	THCudaCheck(cudaGetLastError());
	return 1;
}

static const struct luaL_Reg dcnn_Threshold__[] = { { "Threshold_updateRecOutput", dcnn_Threshold_updateRecOutput }, { NULL, NULL } };

void dcnn_Threshold_init(lua_State *L) {
	luaT_pushmetatable(L, "torch.CudaTensor");
	luaT_registeratname(L, dcnn_Threshold__, "dcnn");
	lua_pop(L, 1);
}
