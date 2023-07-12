#include "PxPhysicsAPI.h"
#include "NpParticleSystem.h"
#include <cstdint>
#include "iostream"
#include "physx_generated.hpp"
#include <malloc.h>
#include "cudamanager/PxCudaContextManager.h"
#include "cudamanager/PxCudaContext.h"

PxDefaultAllocator gAllocator;
PxDefaultErrorCallback gErrorCallback;

struct FilterShaderCallbackInfo {
    PxFilterObjectAttributes attributes0;
    PxFilterObjectAttributes attributes1;
    PxFilterData filterData0;
    PxFilterData filterData1;
    PxPairFlags *pairFlags;
    const void *constantBlock;
    PxU32 constantBlockSize;
};

typedef void (*CollisionCallback)(void *, PxContactPairHeader const *, PxContactPair const *, PxU32);

extern "C" typedef PxU16 (*SimulationShaderFilter)(FilterShaderCallbackInfo *);

struct FilterCallbackData {
    SimulationShaderFilter filter;
    bool call_default_filter_shader_first;
};

PxFilterFlags FilterShaderTrampoline(PxFilterObjectAttributes attributes0,
                                     PxFilterData filterData0,
                                     PxFilterObjectAttributes attributes1,
                                     PxFilterData filterData1,
                                     PxPairFlags &pairFlags,
                                     const void *constantBlock,
                                     PxU32 constantBlockSize) {
    const FilterCallbackData *data = static_cast<const FilterCallbackData *>(constantBlock);

    if (data->call_default_filter_shader_first) {
        // Let the default handler set the pair flags, but ignore the collision filtering
        PxDefaultSimulationFilterShader(attributes0, filterData0, attributes1, filterData1, pairFlags, constantBlock,
                                        constantBlockSize);
    }

    // Get the filter shader from the constant block
    SimulationShaderFilter shaderfilter = data->filter;

    // This is a bit expensive since we're putting things on the stack but with LTO this should optimize OK,
    // and I was having issues with corrupted values when passing by value
    FilterShaderCallbackInfo info{attributes0, attributes1, filterData0, filterData1, &pairFlags, nullptr, 0};

    // We return a u16 since PxFilterFlags is a complex type and C++ wants it to be returned on the stack,
    // but Rust thinks it's simple due to the codegen and wants to return it in EAX.
    return PxFilterFlags{shaderfilter(&info)};
}

using CollisionCallback = void (*)(void *, PxContactPairHeader const *, PxContactPair const *, PxU32);
using TriggerCallback = void (*)(void *, PxTriggerPair const *, PxU32);
using ConstraintBreakCallback = void (*)(void *, PxConstraintInfo const *, PxU32);
using WakeSleepCallback = void (*)(void *, PxActor **const, PxU32, bool);
using AdvanceCallback = void (*)(void *, const PxRigidBody *const *, const PxTransform *const, PxU32);

struct SimulationEventCallbackInfo {
    // Callback for collision events.
    CollisionCallback collisionCallback = nullptr;
    void *collisionUserData = nullptr;
    // Callback for trigger shape events (an object entered or left a trigger shape).
    TriggerCallback triggerCallback = nullptr;
    void *triggerUserData = nullptr;
    // Callback for when a constraint breaks (such as a joint with a force limit)
    ConstraintBreakCallback constraintBreakCallback = nullptr;
    void *constraintBreakUserData = nullptr;
    // Callback for when an object falls asleep or is awoken.
    WakeSleepCallback wakeSleepCallback = nullptr;
    void *wakeSleepUserData = nullptr;
    // Callback to get the next pose early for objects (if flagged with eENABLE_POSE_INTEGRATION_PREVIEW).
    AdvanceCallback advanceCallback = nullptr;
    void *advanceUserData = nullptr;
};

class SimulationEventTrampoline : public PxSimulationEventCallback {
public:
    SimulationEventTrampoline(const SimulationEventCallbackInfo *callbacks) : mCallbacks(*callbacks) {}

    // Collisions
    void onContact(const PxContactPairHeader &pairHeader, const PxContactPair *pairs, PxU32 nbPairs) override {
        if (mCallbacks.collisionCallback) {
            mCallbacks.collisionCallback(mCallbacks.collisionUserData, &pairHeader, pairs, nbPairs);
        }
    }

    // Triggers
    void onTrigger(PxTriggerPair *pairs, PxU32 count) override {
        if (mCallbacks.triggerCallback) {
            mCallbacks.triggerCallback(mCallbacks.triggerUserData, pairs, count);
        }
    }

    // Constraint breaks
    void onConstraintBreak(PxConstraintInfo *constraints, PxU32 count) override {
        if (mCallbacks.constraintBreakCallback) {
            mCallbacks.constraintBreakCallback(mCallbacks.constraintBreakUserData, constraints, count);
        }
    }

    // Wake/Sleep (combined for convenience)
    void onWake(PxActor **actors, PxU32 count) override {
        if (mCallbacks.wakeSleepCallback) {
            mCallbacks.wakeSleepCallback(mCallbacks.wakeSleepUserData, actors, count, true);
        }
    }

    void onSleep(PxActor **actors, PxU32 count) override {
        if (mCallbacks.wakeSleepCallback) {
            mCallbacks.wakeSleepCallback(mCallbacks.wakeSleepUserData, actors, count, false);
        }
    }

    // Advance
    void onAdvance(const PxRigidBody *const *bodyBuffer, const PxTransform *poseBuffer, const PxU32 count) override {
        if (mCallbacks.advanceCallback) {
            mCallbacks.advanceCallback(mCallbacks.advanceUserData, bodyBuffer, poseBuffer, count);
        }
    }

    SimulationEventCallbackInfo mCallbacks;
};

class RaycastFilterCallback : public PxQueryFilterCallback {
public:
    explicit RaycastFilterCallback(PxRigidActor *actor) : mActor(actor) {}

    PxRigidActor *mActor;

    virtual PxQueryHitType::Enum
    preFilter(const PxFilterData &, const PxShape *shape, const PxRigidActor *actor, PxHitFlags &) {
        if (mActor == actor) {
            return PxQueryHitType::eNONE;
        } else {
            return PxQueryHitType::eBLOCK;
        }
    }

    virtual PxQueryHitType::Enum postFilter(const PxFilterData &, const PxQueryHit &) {
        return PxQueryHitType::eNONE;
    }
};

typedef uint32_t (*RaycastHitCallback)(const PxRigidActor *actor, const PxFilterData *filterData, const PxShape *shape,
                                       uint32_t hitFlags, const void *userData);

class RaycastFilterTrampoline : public PxQueryFilterCallback {
public:
    RaycastFilterTrampoline(RaycastHitCallback callback, const void *userdata)
            : mCallback(callback), mUserData(userdata) {}

    RaycastHitCallback mCallback;
    const void *mUserData;

    virtual PxQueryHitType::Enum
    preFilter(const PxFilterData &filterData, const PxShape *shape, const PxRigidActor *actor, PxHitFlags &hitFlags) {
        switch (mCallback(actor, &filterData, shape, (uint32_t) hitFlags, mUserData)) {
            case 0:
                return PxQueryHitType::eNONE;
            case 1:
                return PxQueryHitType::eTOUCH;
            case 2:
                return PxQueryHitType::eBLOCK;
            default:
                return PxQueryHitType::eNONE;
        }
    }

    virtual PxQueryHitType::Enum postFilter(const PxFilterData &, const PxQueryHit &) {
        return PxQueryHitType::eNONE;
    }
};

typedef PxAgain (*RaycastHitProcessTouchesCallback)(const PxRaycastHit *buffer, PxU32 nbHits, void *userdata);

typedef PxAgain (*SweepHitProcessTouchesCallback)(const PxSweepHit *buffer, PxU32 nbHits, void *userdata);

typedef PxAgain (*OverlapHitProcessTouchesCallback)(const PxOverlapHit *buffer, PxU32 nbHits, void *userdata);

typedef void (*HitFinalizeQueryCallback)(void *userdata);

class RaycastHitCallbackTrampoline : public PxRaycastCallback {
public:
    RaycastHitCallbackTrampoline(
            RaycastHitProcessTouchesCallback processTouchesCallback,
            HitFinalizeQueryCallback finalizeQueryCallback,
            PxRaycastHit *touchesBuffer,
            PxU32 numTouches,
            void *userdata)
            : PxRaycastCallback(touchesBuffer, numTouches),
              mProcessTouchesCallback(processTouchesCallback),
              mFinalizeQueryCallback(finalizeQueryCallback),
              mUserData(userdata) {}

    RaycastHitProcessTouchesCallback mProcessTouchesCallback;
    HitFinalizeQueryCallback mFinalizeQueryCallback;
    void *mUserData;

    PxAgain processTouches(const PxRaycastHit *buffer, PxU32 nbHits) override {
        return mProcessTouchesCallback(buffer, nbHits, mUserData);
    }

    void finalizeQuery() override {
        mFinalizeQueryCallback(mUserData);
    }
};

class SweepHitCallbackTrampoline : public PxSweepCallback {
public:
    SweepHitCallbackTrampoline(
            SweepHitProcessTouchesCallback processTouchesCallback,
            HitFinalizeQueryCallback finalizeQueryCallback,
            PxSweepHit *touchesBuffer,
            PxU32 numTouches,
            void *userdata)
            : PxSweepCallback(touchesBuffer, numTouches),
              mProcessTouchesCallback(processTouchesCallback),
              mFinalizeQueryCallback(finalizeQueryCallback),
              mUserData(userdata) {}

    SweepHitProcessTouchesCallback mProcessTouchesCallback;
    HitFinalizeQueryCallback mFinalizeQueryCallback;
    void *mUserData;

    PxAgain processTouches(const PxSweepHit *buffer, PxU32 nbHits) override {
        return mProcessTouchesCallback(buffer, nbHits, mUserData);
    }

    void finalizeQuery() override {
        mFinalizeQueryCallback(mUserData);
    }
};

class OverlapHitCallbackTrampoline : public PxOverlapCallback {
public:
    OverlapHitCallbackTrampoline(
            OverlapHitProcessTouchesCallback processTouchesCallback,
            HitFinalizeQueryCallback finalizeQueryCallback,
            PxOverlapHit *touchesBuffer,
            PxU32 numTouches,
            void *userdata)
            : PxOverlapCallback(touchesBuffer, numTouches),
              mProcessTouchesCallback(processTouchesCallback),
              mFinalizeQueryCallback(finalizeQueryCallback),
              mUserData(userdata) {}

    OverlapHitProcessTouchesCallback mProcessTouchesCallback;
    HitFinalizeQueryCallback mFinalizeQueryCallback;
    void *mUserData;

    PxAgain processTouches(const PxOverlapHit *buffer, PxU32 nbHits) override {
        return mProcessTouchesCallback(buffer, nbHits, mUserData);
    }

    void finalizeQuery() override {
        mFinalizeQueryCallback(mUserData);
    }
};

typedef void *(*AllocCallback)(uint64_t size, const char *typeName, const char *filename, int line, void *userdata);

typedef void (*DeallocCallback)(void *ptr, void *userdata);

class CustomAllocatorTrampoline : public PxAllocatorCallback {
public:
    CustomAllocatorTrampoline(AllocCallback allocCb, DeallocCallback deallocCb, void *userdata)
            : mAllocCallback(allocCb), mDeallocCallback(deallocCb), mUserData(userdata) {}

    void *allocate(size_t size, const char *typeName, const char *filename, int line) {
        return mAllocCallback((uint64_t) size, typeName, filename, line, mUserData);
    }

    virtual void deallocate(void *ptr) {
        mDeallocCallback(ptr, mUserData);
    }

private:
    AllocCallback mAllocCallback;
    DeallocCallback mDeallocCallback;
public:
    void *mUserData;
};

typedef void *(*ZoneStartCallback)(const char *typeName, bool detached, uint64_t context, void *userdata);

typedef void  (*ZoneEndCallback)(void *profilerData, const char *typeName, bool detached, uint64_t context,
                                 void *userdata);

class CustomProfilerTrampoline : public PxProfilerCallback {
public:
    CustomProfilerTrampoline(ZoneStartCallback startCb, ZoneEndCallback endCb, void *userdata)
            : mStartCallback(startCb), mEndCallback(endCb), mUserData(userdata) {
    }

    virtual void *zoneStart(const char *eventName, bool detached, uint64_t contextId) override {
        return mStartCallback(eventName, detached, contextId, mUserData);
    }

    virtual void zoneEnd(void *profilerData, const char *eventName, bool detached, uint64_t contextId) override {
        return mEndCallback(profilerData, eventName, detached, contextId, mUserData);
    }

private:
    ZoneStartCallback mStartCallback;
    ZoneEndCallback mEndCallback;
public:
    void *mUserData;
};

using ErrorCallback = void (*)(int code, const char *message, const char *file, int line, void *userdata);

class ErrorTrampoline : public PxErrorCallback {
public:
    ErrorTrampoline(ErrorCallback errorCb, void *userdata)
            : mErrorCallback(errorCb), mUserdata(userdata) {}

    void reportError(PxErrorCode::Enum code, const char *message, const char *file, int line) override {
        mErrorCallback(code, message, file, line, mUserdata);
    }

private:
    ErrorCallback mErrorCallback = nullptr;
    void *mUserdata = nullptr;
};

using AssertHandler = void (*)(const char *expr, const char *file, int line, bool *should_ignore, void *userdata);

class AssertTrampoline : public PxAssertHandler {
public:
    AssertTrampoline(AssertHandler onAssert, void *userdata)
            : mAssertHandler(onAssert), mUserdata(userdata) {}

    virtual void operator()(const char *exp, const char *file, int line, bool &ignore) override final {
        mAssertHandler(exp, file, line, &ignore, mUserdata);
    }

private:
    AssertHandler mAssertHandler = nullptr;
    void *mUserdata = nullptr;
};

extern "C"
{
    void setU32At(PxU32 *base, int index, PxU32 value) {
        base[index] = value;
    }

    PxU32 getU32At(const PxU32 *base, int index) {
        return base[index];
    }

    void setVec4At(void *base, int index, PxVec4 value) {
        //base[index] = value;
        static_cast<physx::PxVec4 *>(base)[index] = value;
    }

    PxVec4 *getVec4At(PxVec4 *base, int index) {
        return &base[index];
    }

    void vec4SetX(PxVec4 *_address, float value) {
        PxVec4 *_self = (PxVec4 *) _address;
        _self->x = value;
    }

    void vec4SetY(PxVec4 *_address, float value) {
        PxVec4 *_self = (PxVec4 *) _address;
        _self->y = value;
    }

    void vec4SetZ(PxVec4 *_address, float value) {
        PxVec4 *_self = (PxVec4 *) _address;
        _self->z = value;
    }

    void vec4SetW(PxVec4 *_address, float value) {
        PxVec4 *_self = (PxVec4 *) _address;
        _self->w = value;
    }

    float vec4GetX(PxVec4 *_address) {
        PxVec4 *_self = (PxVec4 *) _address;
        return _self->x;
    }

    float vec4GetY(PxVec4 *_address) {
        PxVec4 *_self = (PxVec4 *) _address;
        return _self->y;
    }

    float vec4GetZ(PxVec4 *_address) {
        PxVec4 *_self = (PxVec4 *) _address;
        return _self->z;
    }

    float vec4GetW(PxVec4 *_address) {
        PxVec4 *_self = (PxVec4 *) _address;
        return _self->w;
    }

    void PxParticleBufferDesc_setPhases(PxParticleBufferDesc *_address, PxU32 *value) {
        PxParticleBufferDesc *_self = (PxParticleBufferDesc *) _address;
        _self->phases = (PxU32 *) value;
    }

    void PxParticleBufferDesc_setVelocities(PxParticleBufferDesc *_address, PxVec4 *value) {
        PxParticleBufferDesc *_self = (PxParticleBufferDesc *) _address;
        _self->velocities = (PxVec4 *) value;
    }

    void PxParticleBufferDesc_setPositions(PxParticleBufferDesc *_address, PxVec4 *value) {
        PxParticleBufferDesc *_self = (PxParticleBufferDesc *) _address;
        _self->positions = (PxVec4 *) value;
    }

    PxU32 *alloc_pinned_host_buffer_pxu32(PxCudaContextManager *cudaContextManager, PxU32 numElements) {
        return cudaContextManager->allocPinnedHostBuffer<PxU32>(numElements);
    }

    PxVec4 *alloc_pinned_host_buffer_pxvec4(PxCudaContextManager *cudaContextManager, PxU32 numElements) {
        return cudaContextManager->allocPinnedHostBuffer<PxVec4>(numElements);
    }

    PxCudaContextManager *physx_create_cuda_context_manager(PxFoundation &foundation, const PxCudaContextManagerDesc &desc,
                                                            PxProfilerCallback *profilerCallback) {
        return PxCreateCudaContextManager(foundation, desc, profilerCallback);
    }

    PxFoundation *physx_create_foundation() {
        return PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
    }

    PxFoundation *physx_create_foundation_with_alloc(PxAllocatorCallback *allocator) {
        return PxCreateFoundation(PX_PHYSICS_VERSION, *allocator, gErrorCallback);
    }

    // fixme[tolsson]: this might be iffy on Windows with DLLs if we have multiple packages
    // linking against the raw interface
    PxAllocatorCallback *get_default_allocator() {
        return &gAllocator;
    }

    // fixme[tolsson]: this might be iffy on Windows with DLLs if we have multiple packages
    // linking against the raw interface
    PxErrorCallback *get_default_error_callback() {
        return &gErrorCallback;
    }

    PxPhysics *physx_create_physics(PxFoundation *foundation) {
        return PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale(), true, nullptr, nullptr);
    }

    PxQueryFilterCallback *create_raycast_filter_callback(PxRigidActor *actor_to_ignore) {
        return new RaycastFilterCallback(actor_to_ignore);
    }

    PxQueryFilterCallback *create_raycast_filter_callback_func(RaycastHitCallback callback, void *userData) {
        return new RaycastFilterTrampoline(callback, userData);
    }

    PxRaycastCallback *create_raycast_buffer() {
        return new PxRaycastBuffer;
    }

    PxSweepCallback *create_sweep_buffer() {
        return new PxSweepBuffer;
    }

    PxOverlapCallback *create_overlap_buffer() {
        return new PxOverlapBuffer;
    }

    PxRaycastCallback *create_raycast_callback(
            RaycastHitProcessTouchesCallback process_touches_callback,
            HitFinalizeQueryCallback finalize_query_callback,
            PxRaycastHit *touchesBuffer,
            PxU32 numTouches,
            void *userdata
    ) {
        return new RaycastHitCallbackTrampoline(
                process_touches_callback, finalize_query_callback, touchesBuffer, numTouches, userdata);
    }

    void delete_raycast_callback(PxRaycastCallback *callback) {
        delete callback;
    }

    void delete_sweep_callback(PxSweepCallback *callback) {
        delete callback;
    }

    void delete_overlap_callback(PxOverlapCallback *callback) {
        delete callback;
    }

    PxSweepCallback *create_sweep_callback(
            SweepHitProcessTouchesCallback process_touches_callback,
            HitFinalizeQueryCallback finalize_query_callback,
            PxSweepHit *touchesBuffer,
            PxU32 numTouches,
            void *userdata
    ) {
        return new SweepHitCallbackTrampoline(
                process_touches_callback, finalize_query_callback, touchesBuffer, numTouches, userdata
        );
    }

    PxOverlapCallback *create_overlap_callback(
            OverlapHitProcessTouchesCallback process_touches_callback,
            HitFinalizeQueryCallback finalize_query_callback,
            PxOverlapHit *touchesBuffer,
            PxU32 numTouches,
            void *userdata
    ) {
        return new OverlapHitCallbackTrampoline(
                process_touches_callback, finalize_query_callback, touchesBuffer, numTouches, userdata
        );
    }

    PxAllocatorCallback *create_alloc_callback(
            AllocCallback alloc_callback,
            DeallocCallback dealloc_callback,
            void *userdata
    ) {
        return new CustomAllocatorTrampoline(alloc_callback, dealloc_callback, userdata);
    }

    void *get_alloc_callback_user_data(PxAllocatorCallback *allocator) {
        CustomAllocatorTrampoline *trampoline = static_cast<CustomAllocatorTrampoline *>(allocator);
        return trampoline->mUserData;
    }

    PxProfilerCallback *create_profiler_callback(
            ZoneStartCallback zone_start_callback,
            ZoneEndCallback zone_end_callback,
            void *userdata
    ) {
        return new CustomProfilerTrampoline(zone_start_callback, zone_end_callback, userdata);
    }

    PxErrorCallback *create_error_callback(
            ErrorCallback error_callback,
            void *userdata
    ) {
        return new ErrorTrampoline(error_callback, userdata);
    }


    PxAssertHandler *create_assert_handler(
            AssertHandler on_assert,
            void *userdata
    ) {
        return new AssertTrampoline(on_assert, userdata);
    }

    void *get_default_simulation_filter_shader() {
        return (void *) PxDefaultSimulationFilterShader;
    }

    PxSimulationEventCallback *create_simulation_event_callbacks(const SimulationEventCallbackInfo *callbacks) {
        SimulationEventTrampoline *trampoline = new SimulationEventTrampoline(callbacks);
        return static_cast<PxSimulationEventCallback *>(trampoline);
    }

    SimulationEventCallbackInfo *get_simulation_event_info(PxSimulationEventCallback *callback) {
        SimulationEventTrampoline *trampoline = static_cast<SimulationEventTrampoline *>(callback);
        return &trampoline->mCallbacks;
    }

    void destroy_simulation_event_callbacks(PxSimulationEventCallback *callback) {
        SimulationEventTrampoline *trampoline = static_cast<SimulationEventTrampoline *>(callback);
        delete trampoline;
    }

    void enable_custom_filter_shader(PxSceneDesc *desc, SimulationShaderFilter filter,
                                     uint32_t call_default_filter_shader_first) {
        /* Note: This is a workaround to PhysX copying the filter data */
        static FilterCallbackData filterShaderData = {
                filter,
                call_default_filter_shader_first != 0
        };
        desc->filterShader = FilterShaderTrampoline;
        // printf("Setting pointer to %p\n", filter);
        desc->filterShaderData = (void *) &filterShaderData;
        desc->filterShaderDataSize = sizeof(FilterCallbackData);
    }

    // Not generated, used only for testing and examples!
    void PxAssertHandler_opCall_mut(physx_PxErrorCallback_Pod *self__pod, char const *expr, char const *file, int32_t line,
                                    bool *ignore) {
        physx::PxAssertHandler *self_ = reinterpret_cast<physx::PxAssertHandler *>(self__pod);
        (*self_)(expr, file, line, *ignore);
    } ;

    physx_PxVec4_Pod *
    PxParticleAndDiffuseBuffer_getPositionInvMasses(physx_PxParticleAndDiffuseBuffer_Pod const *self__pod) {
        physx::PxParticleAndDiffuseBuffer const *self_ = reinterpret_cast<physx::PxParticleAndDiffuseBuffer const *>(self__pod);
        physx::PxVec4 *return_val = self_->getPositionInvMasses();
        auto return_val_pod = reinterpret_cast<physx_PxVec4_Pod *>(return_val);
        return return_val_pod;
    }

    physx_PxPBDParticleSystem_Pod *PxScene_getPBDParticleSystems(PxCudaContextManager *cudaContextManager, physx_PxScene_Pod const *self__pod, int32_t type_pod, uint32_t bufferSize, uint32_t startIndex) {
        PxParticleSystem **userBuffer_pod;
        userBuffer_pod = (PxParticleSystem **) malloc(sizeof(PxParticleSystem * ));

        physx::PxScene const *self_ = reinterpret_cast<physx::PxScene const *>(self__pod);
        auto type = static_cast<physx::PxParticleSolverType::Enum>(type_pod);

        uint32_t return_val = self_->getParticleSystems(type, userBuffer_pod, bufferSize, startIndex);
        return reinterpret_cast<physx_PxPBDParticleSystem_Pod *>(userBuffer_pod[0]);
    }

    physx_PxParticleBuffer_Pod* PxParticleSystem_getParticleBuffer(PxParticleSystem *particle_system) {
        NpPBDParticleSystem *npParticleSystem = static_cast<NpPBDParticleSystem *>(particle_system);
        Sc::ParticleSystemCore &sc_core = npParticleSystem->getCore();
        Sc::ParticleSystemShapeCore& shapeCore = sc_core.getShapeCore();
        Dy::ParticleSystemCore& core = shapeCore.getLLCore();

        //std::cout << <<std::endl;
        return reinterpret_cast<physx_PxParticleBuffer_Pod*>(core.mParticleAndDiffuseBuffers[0]);
        //std::cout << core.mParticleBuffers.size()<<std::endl;
    }

    PxVec4* getVec4ArrayFromGPU(PxCudaContextManager *cudaContextManager, int array_size, PxVec4* pointer) {
        PxVec4* buffer = new PxVec4[array_size];

        cudaContextManager->acquireContext();
        PxCudaContext* cuda_context = cudaContextManager->getCudaContext();
        cuda_context->memcpyDtoH(buffer, reinterpret_cast<CUdeviceptr>(pointer), sizeof(PxVec4) * array_size);
        cudaContextManager->releaseContext();
        return buffer;
    }

    void freeVec4Array(PxCudaContextManager *cudaContextManager, PxVec4* pointer) {
        cudaContextManager->acquireContext();
        PxCudaContext* cuda_context = cudaContextManager->getCudaContext();
        cuda_context->memFreeHost(pointer);
        cudaContextManager->releaseContext();
    }

}
