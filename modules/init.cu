#include <luaT.h>
#include <THC.h>
#include "TH.h"
#include <THLogAdd.h>

#include "utils.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libdcnn(lua_State *L);

int luaopen_libdcnn(lua_State *L) {
	lua_newtable(L);
	lua_pushvalue(L, -1);
	lua_setglobal(L, "dcnn");

	dcnn_DualSpatialMaxPooling_init(L);
	dcnn_DualSpatialMaxUnpooling_init(L);

	return 1;
}
