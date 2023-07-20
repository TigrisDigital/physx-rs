// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2023 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

#ifndef PX_VEC3_H
#define PX_VEC3_H

/** \addtogroup foundation
@{
*/

#include "foundation/PxMath.h"

#if !PX_DOXYGEN
namespace physx
{
#endif

/**
\brief 3 Element vector class.

This is a 3-dimensional vector class with public data members.
*/
class PxVec3
{
  public:

	/**
	\brief default constructor leaves data uninitialized.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3()
	{
	}

	/**
	\brief zero constructor.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3(PxZERO) : x(float(0.0)), y(float(0.0)), z(float(0.0))
	{
	}

	/**
	\brief Assigns scalar parameter to all elements.

	Useful to initialize to zero or one.

	\param[in] a Value to assign to elements.
	*/
	explicit PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3(float a) : x(a), y(a), z(a)
	{
	}

	/**
	\brief Initializes from 3 scalar parameters.

	\param[in] nx Value to initialize X component.
	\param[in] ny Value to initialize Y component.
	\param[in] nz Value to initialize Z component.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3(float nx, float ny, float nz) : x(nx), y(ny), z(nz)
	{
	}

	/**
	\brief Copy ctor.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3(const PxVec3& v) : x(v.x), y(v.y), z(v.z)
	{
	}

	// Operators

	/**
	\brief Assignment operator
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3& operator=(const PxVec3& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		return *this;
	}

	/**
	\brief element access
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE float& operator[](unsigned int index)
	{
		PX_ASSERT(index <= 2);
		return reinterpret_cast<float*>(this)[index];
	}

	/**
	\brief element access
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE const float& operator[](unsigned int index) const
	{
		PX_ASSERT(index <= 2);
		return reinterpret_cast<const float*>(this)[index];
	}

	/**
	\brief returns true if the two vectors are exactly equal.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE bool operator==(const PxVec3& v) const
	{
		return x == v.x && y == v.y && z == v.z;
	}

	/**
	\brief returns true if the two vectors are not exactly equal.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE bool operator!=(const PxVec3& v) const
	{
		return x != v.x || y != v.y || z != v.z;
	}

	/**
	\brief tests for exact zero vector
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE bool isZero() const
	{
		return x == float(0.0) && y == float(0.0) && z == float(0.0);
	}

	/**
	\brief returns true if all 3 elems of the vector are finite (not NAN or INF, etc.)
	*/
	PX_CUDA_CALLABLE PX_INLINE bool isFinite() const
	{
		return PxIsFinite(x) && PxIsFinite(y) && PxIsFinite(z);
	}

	/**
	\brief is normalized - used by API parameter validation
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE bool isNormalized() const
	{
		const float unitTolerance = float(1e-4);	// PT: do we need a different epsilon for float & double?
		return isFinite() && PxAbs(magnitude() - float(1.0)) < unitTolerance;
	}

	/**
	\brief returns the squared magnitude

	Avoids calling PxSqrt()!
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE float magnitudeSquared() const
	{
		return x * x + y * y + z * z;
	}

	/**
	\brief returns the magnitude
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE float magnitude() const
	{
		return PxSqrt(magnitudeSquared());
	}

	/**
	\brief negation
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 operator-() const
	{
		return PxVec3(-x, -y, -z);
	}

	/**
	\brief vector addition
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 operator+(const PxVec3& v) const
	{
		return PxVec3(x + v.x, y + v.y, z + v.z);
	}

	/**
	\brief vector difference
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 operator-(const PxVec3& v) const
	{
		return PxVec3(x - v.x, y - v.y, z - v.z);
	}

	/**
	\brief scalar post-multiplication
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 operator*(float f) const
	{
		return PxVec3(x * f, y * f, z * f);
	}

	/**
	\brief scalar division
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 operator/(float f) const
	{
		f = float(1.0) / f;
		return PxVec3(x * f, y * f, z * f);
	}

	/**
	\brief vector addition
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3& operator+=(const PxVec3& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	/**
	\brief vector difference
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3& operator-=(const PxVec3& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	/**
	\brief scalar multiplication
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3& operator*=(float f)
	{
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}
	/**
	\brief scalar division
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3& operator/=(float f)
	{
		f = float(1.0) / f;
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}

	/**
	\brief returns the scalar product of this and other.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE float dot(const PxVec3& v) const
	{
		return x * v.x + y * v.y + z * v.z;
	}

	/**
	\brief cross product
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 cross(const PxVec3& v) const
	{
		return PxVec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}

	/** returns a unit vector */
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 getNormalized() const
	{
		const float m = magnitudeSquared();
		return m > float(0.0) ? *this * PxRecipSqrt(m) : PxVec3(float(0));
	}

	/**
	\brief normalizes the vector in place
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE float normalize()
	{
		const float m = magnitude();
		if(m > float(0.0))
			*this /= m;
		return m;
	}

	/**
	\brief normalizes the vector in place. Does nothing if vector magnitude is under PX_NORMALIZATION_EPSILON.
	Returns vector magnitude if >= PX_NORMALIZATION_EPSILON and 0.0f otherwise.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE float normalizeSafe()
	{
		const float mag = magnitude();
		if(mag < PX_NORMALIZATION_EPSILON)	// PT: do we need a different epsilon for float & double?
			return float(0.0);
		*this *= float(1.0) / mag;
		return mag;
	}

	/**
	\brief normalizes the vector in place. Asserts if vector magnitude is under PX_NORMALIZATION_EPSILON.
	returns vector magnitude.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE float normalizeFast()
	{
		const float mag = magnitude();
		PX_ASSERT(mag >= PX_NORMALIZATION_EPSILON);	// PT: do we need a different epsilon for float & double?
		*this *= float(1.0) / mag;
		return mag;
	}

	/**
	\brief a[i] * b[i], for all i.
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 multiply(const PxVec3& a) const
	{
		return PxVec3(x * a.x, y * a.y, z * a.z);
	}

	/**
	\brief element-wise minimum
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 minimum(const PxVec3& v) const
	{
		return PxVec3(PxMin(x, v.x), PxMin(y, v.y), PxMin(z, v.z));
	}

	/**
	\brief returns MIN(x, y, z);
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE float minElement() const
	{
		return PxMin(x, PxMin(y, z));
	}

	/**
	\brief element-wise maximum
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 maximum(const PxVec3& v) const
	{
		return PxVec3(PxMax(x, v.x), PxMax(y, v.y), PxMax(z, v.z));
	}

	/**
	\brief returns MAX(x, y, z);
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE float maxElement() const
	{
		return PxMax(x, PxMax(y, z));
	}

	/**
	\brief returns absolute values of components;
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3 abs() const
	{
		return PxVec3(PxAbs(x), PxAbs(y), PxAbs(z));
	}

	float	x, y, z;
};

PX_CUDA_CALLABLE static PX_FORCE_INLINE PxVec3 operator*(float f, const PxVec3& v)
{
	return PxVec3(f * v.x, f * v.y, f * v.z);
}

//! A padded version of PxVec3, to safely load its data using SIMD
class PxVec3Padded : public PxVec3
{
	public:
	PX_FORCE_INLINE	PxVec3Padded()								{}
	PX_FORCE_INLINE	~PxVec3Padded()								{}
	PX_FORCE_INLINE	PxVec3Padded(const PxVec3& p) : PxVec3(p)	{}
	PX_FORCE_INLINE	PxVec3Padded(float f) : PxVec3(f)			{}

	/**
	\brief Assignment operator.
	To fix this:
	error: definition of implicit copy assignment operator for 'PxVec3Padded' is deprecated because it has a user-declared destructor [-Werror,-Wdeprecated]
	*/
	PX_CUDA_CALLABLE PX_FORCE_INLINE PxVec3Padded& operator=(const PxVec3Padded& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		return *this;
	}

	PxU32	padding;
};
PX_COMPILE_TIME_ASSERT(sizeof(PxVec3Padded) == 16);

typedef PxVec3Padded	PxVec3p;

#if !PX_DOXYGEN
} // namespace physx
#endif

/** @} */
#endif
