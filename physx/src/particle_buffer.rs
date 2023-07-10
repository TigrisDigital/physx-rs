use crate::owner::Owner;
use crate::traits::Class;

#[repr(transparent)]
pub struct PxParticleBuffer {
    obj: *mut physx_sys::PxParticleBuffer
}

unsafe impl<P> Class<P> for PxParticleBuffer
    where
        physx_sys::PxParticleBuffer: Class<P>
{
    fn as_ptr(&self) -> *const P {
        self.obj as *const P
    }

    fn as_mut_ptr(&mut self) -> *mut P {
        self.obj as *mut P
    }
}

unsafe impl Send for PxParticleBuffer {}
unsafe impl Sync for PxParticleBuffer {}

impl PxParticleBuffer {
    pub fn new(particle_buffer: *mut physx_sys::PxParticleBuffer) -> PxParticleBuffer {
        Self {
            obj: particle_buffer
        }
    }

    pub unsafe fn from_raw(ptr: *mut physx_sys::PxParticleBuffer) -> Option<Owner<Self>> {
        // userData is initialized by the descriptor.
        unsafe { Owner::from_raw(   ptr as *mut Self) }
    }
}
