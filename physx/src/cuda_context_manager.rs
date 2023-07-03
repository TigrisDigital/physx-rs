use physx_sys::PxCudaContextManager_release_mut;
use crate::{
    traits::Class,
    owner::Owner
};

#[repr(transparent)]
pub struct PxCudaContextManager {
    obj: physx_sys::PxCudaContextManager
}

// impl Drop for PxCudaContextManager {
//     fn drop(&mut self) {
//         unsafe {
//             PxCudaContextManager_release_mut(self.as_mut_ptr())
//         }
//     }
// }

unsafe impl<P> Class<P> for PxCudaContextManager
where
    physx_sys::PxCudaContextManager: Class<P>
{
    fn as_ptr(&self) -> *const P {
        self.obj.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut P {
        self.obj.as_mut_ptr()
    }
}

unsafe impl Send for PxCudaContextManager {}
unsafe impl Sync for PxCudaContextManager {}

impl PxCudaContextManager {
    pub fn new(context_manager: physx_sys::PxCudaContextManager) -> PxCudaContextManager {
        Self {
            obj: context_manager
        }
    }

    pub unsafe fn from_raw(ptr: *mut physx_sys::PxCudaContextManager) -> Option<Owner<Self>> {
        // userData is initialized by the descriptor.
        unsafe { Owner::from_raw(ptr as *mut Self) }
    }
}