#ifndef DCNN_UTILS_H
#define DCNN_UTILS_H

extern "C"
{
#include <lua.h>
}
#include <luaT.h>
#include <THC.h>

THCState* getCutorchState(lua_State* L);

void dcnn_DualSpatialMaxPooling_init(lua_State *L);
void dcnn_DualSpatialMaxUnpooling_init(lua_State *L);

void dcnn_Threshold_init(lua_State *L);


#endif
