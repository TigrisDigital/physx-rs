use crate::owner::Owner;
use crate::traits::Class;

#[repr(transparent)]
pub struct PxParticleAndDiffuseBuffer {
    obj: *mut physx_sys::PxParticleAndDiffuseBuffer
}

unsafe impl<P> Class<P> for PxParticleAndDiffuseBuffer
    where
        physx_sys::PxParticleAndDiffuseBuffer: Class<P>
{
    fn as_ptr(&self) -> *const P {
        self.obj as *const P
    }

    fn as_mut_ptr(&mut self) -> *mut P {
        self.obj as *mut P
    }
}

unsafe impl Send for PxParticleAndDiffuseBuffer {}
unsafe impl Sync for PxParticleAndDiffuseBuffer {}

impl PxParticleAndDiffuseBuffer {
    pub fn new(particle_buffer: *mut physx_sys::PxParticleAndDiffuseBuffer) -> PxParticleAndDiffuseBuffer {
        Self {
            obj: particle_buffer
        }
    }

    pub unsafe fn from_raw(ptr: *mut physx_sys::PxParticleAndDiffuseBuffer) -> Option<Owner<Self>> {
        // userData is initialized by the descriptor.
        unsafe { Owner::from_raw(   ptr as *mut Self) }
    }
}
