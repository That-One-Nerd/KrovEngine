#include <iostream>	
#include "opencv2/opencv.hpp"
#include <fstream>
#include <strstream>
#include <math.h>
#include <Windows.h>
#include <vector>
#include <algorithm>
#include <strstream>
#include <filesystem>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse.h>
namespace KrovEngine
{
	
	__host__ __device__ float fast_inverse_sqrt(float number)
	{
		long i;
		float x2, y;
		const float threehalfs = 1.5F;

		x2 = number * 0.5F;
		y = number;
		i = *(long*)&y;  
		i = 0x5f3759df - (i >> 1);
		y = *(float*)&i;
		y = y * (threehalfs - (x2 * y * y));
		return y;
	}

	template<class T>
	class Vector3
	{
	public:
		T x;
		T y;
		T z;
		__host__ __device__ Vector3(T x, T y, T z)
		{
			this->x = x;
			this->y = y;
			this->z = z;
		}
		__host__ __device__ Vector3(T x)
		{
			this->x = x;
			this->y = x;
			this->z = x;
		}
		__host__ __device__ Vector3()
		{
			// Default
			this->x = (T)0;
			this->y = (T)0;
			this->z = (T)0;
		}
		__host__ __device__ void fill(T(*gen)(int idx))
		{
			this->x = gen(0);
			this->y = gen(1);
			this->z = gen(2);
		}
		__host__ __device__ T dot(Vector3<T> other)
		{
			return this->x * other.x + this->y * other.y + this->z * other.z;
		}
		__host__ __device__ Vector3<T> cross(Vector3<T> other)
		{
			return Vector3<T>(
				this->y * other.z - this->z * other.y,
				this->z * other.x - this->x * other.z,
				this->x * other.y - this->y * other.x
			);
		}
		__host__ __device__ Vector3<T>& operator-(const Vector3<T> other)
		{
			return Vector3<T>(this->x - other.x, this->y - other.y, this->z - other.z);
		}
		__host__ __device__ Vector3<T>& operator*(const Vector3<T> other)
		{
			return Vector3<T>(this->x * other.x, this->y * other.y, this->z * other.z);
		}
		__host__ __device__ Vector3<T>& operator/(const Vector3<T> other)
		{
			return Vector3<T>(this->x / other.x, this->y / other.y, this->z / other.z);
		}
		__host__ __device__ Vector3<T>& operator+(const Vector3<T> other)
		{
			return Vector3<T>(this->x + other.x, this->y + other.y, this->z + other.z);
		}
		__host__ __device__ float magnitude()
		{
			return sqrtf(this->x * this->x + this->y * this->y + this->z * this->z);
		}
		__host__ __device__ Vector3<T> normalize()
		{
			float inv_mag = fast_inverse_sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
			Vector3<T> res;
			res.x = this->x * inv_mag;
			res.y = this->y * inv_mag;
			res.z = this->z * inv_mag;
			return res;
		}
	};
	template<class T>
	class Vector4
	{
	public:
		T x;
		T y;
		T z;
		T w;
		__host__ __device__ Vector4(T x, T y, T z, T w)
		{
			this->x = x;
			this->y = y;
			this->z = z;
			this->w = w;
		}
		__host__ __device__ Vector4(T x)
		{
			this->x = x;
			this->y = x;
			this->z = x;
			this->w = x;
		}
		__host__ __device__ Vector4()
		{
			// Default
		}
		__host__ __device__ void fill(T(*gen)(int idx))
		{
			this->x = gen(0);
			this->y = gen(1);
			this->z = gen(2);
			this->w = gen(2);
		}
		__host__ __device__ T dot(Vector4<T> other)
		{
			return this->x * other.x + this->y * other.y + this->z * other.z + this->w * other.w;
		}
		__host__ __device__ Vector4<T>& operator-(const Vector4<T> other)
		{
			return Vector3<T>(this->x - other.x, this->y - other.y, this->z - other.z, this->w - other.w);
		}
		__host__ __device__ Vector4<T>& operator*(const Vector4<T> other)
		{
			return Vector3<T>(this->x * other.x, this->y * other.y, this->z * other.z, this->w * other.w);
		}
		__host__ __device__ Vector4<T>& operator/(const Vector4<T> other)
		{
			return Vector3<T>(this->x / other.x, this->y / other.y, this->z / other.z, this->w / other.w);
		}
		__host__ __device__ Vector4<T>& operator+(const Vector4<T> other)
		{
			return Vector4<T>(this->x + other.x, this->y + other.y, this->z + other.z, this->w + other.w);
		}
		__host__ __device__ float magnitude()
		{
			return sqrtf(this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w);
		}
		__host__ __device__ Vector4<T> normalize()
		{
			float inv_mag = fast_inverse_sqrt(this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w);
			Vector4<T> res;
			res.x = this->x * inv_mag;
			res.y = this->y * inv_mag;
			res.z = this->z * inv_mag;
			res.w = this->w * inv_mag;
			return res;
		}
	};
	template<class T>
	class Vector2
	{
	public:
		T x;
		T y;
		__host__ __device__ Vector2(T x, T y)
		{
			this->x = x;
			this->y = y;
		}
		__host__ __device__ Vector2(T x)
		{
			this->x = x;
			this->y = x;
		}
		__host__ __device__ Vector2()
		{
			// Default
			this->x = (T)0;
			this->y = (T)0;
		}
		__host__ __device__ void fill(T(*gen)(int idx))
		{
			this->x = gen(0);
			this->y = gen(1);
		}
		__host__ __device__ T dot(Vector2<T> other)
		{
			return this->x * other.x + this->y * other.y;
		}
		__host__ __device__ Vector2<T>& operator-(const Vector2<T> other)
		{
			return Vector2<T>(this->x - other.x, this->y - other.y);
		}
		__host__ __device__ Vector2<T>& operator*(const Vector2<T> other)
		{
			return Vector2<T>(this->x * other.x, this->y * other.y);
		}
		__host__ __device__ Vector2<T>& operator/(const Vector2<T> other)
		{
			return Vector2<T>(this->x / other.x, this->y / other.y);
		}
		__host__ __device__ Vector2<T>& operator+(const Vector2<T> other)
		{
			return Vector2<T>(this->x + other.x, this->y + other.y);
		}
		__host__ __device__ float magnitude()
		{
			return sqrtf(this->x * this->x + this->y * this->y);
		}
		__host__ __device__ Vector2<T> normalize()
		{
			float inv_mag = fast_inverse_sqrt(this->x * this->x + this->y * this->y);
			Vector2<T> res;
			res.x = this->x * inv_mag;
			res.y = this->y * inv_mag;
			return res;
		}
	};
	class Texture2D
	{
	public:
		float data[512][512][3];
		float normal_data[512][512][3];
		bool normal;
		bool albedo;
		__host__ __device__ Texture2D(bool normal, bool albedo)
		{
			this->normal = normal;
			this->albedo = albedo;
		}
		__host__ __device__ Texture2D()
		{
			// Default
		}
	};
	void texture_fill(Texture2D& tex, const char* texture_directory, bool write_normals = false)
	{
		cv::Mat cv_tex = cv::imread(texture_directory);
		cv::resize(cv_tex, cv_tex, cv::Size(512, 512));
		cv_tex.convertTo(cv_tex, CV_8UC3);
		for (int y = 0;y < 512;y++)
		{
			for (int x = 0;x < 512;x++)
			{
				cv::Vec3b& at = cv_tex.at<cv::Vec3b>(cv::Point(y, x));
				if (!write_normals)
				{
					tex.data[y][x][0] = at.val[0];
					tex.data[y][x][1] = at.val[1];
					tex.data[y][x][2] = at.val[2];
				}
				else
				{
					tex.normal_data[y][x][0] = (((float)at.val[0]) / 255.0f - 0.5f) * 2.0f;
					tex.normal_data[y][x][1] = (((float)at.val[1]) / 255.0f - 0.5f) * 2.0f;
					tex.normal_data[y][x][2] = (((float)at.val[2]) / 255.0f - 0.5f) * 2.0f;
				}
			}
		}
	}
	class Triangle
	{
	public:
		Vector3<float> vertices[3];
		Vector3<int> color;
		Vector3<float> normal;
		int texture_idx;
		bool textured = false;
		bool glass = false;
		bool reflective = false;
		float reflective_index;
		__host__ __device__ Triangle(Vector3<float> p1, Vector3<float> p2, Vector3<float> p3)
		{
			this->vertices[0] = p1;
			this->vertices[1] = p2;
			this->vertices[2] = p3;
		}
		__host__ __device__ Triangle()
		{
			// Default
		}
		__host__ __device__ void set_vertices(Vector3<float> p1, Vector3<float> p2, Vector3<float> p3)
		{
			this->vertices[0] = p1;
			this->vertices[1] = p2;
			this->vertices[2] = p3;
		}
		__host__ __device__ void calc_normal()
		{
			this->normal = ((this->vertices[2] - this->vertices[0]).cross(this->vertices[1] - this->vertices[0])).normalize();
		}
	};
	
	class Ray
	{
	public:
		Vector3<float> origin;
		Vector3<float> direction;
		Vector3<int> color;
		float u = 0.0f;
		float v = 0.0f;
		__host__ __device__ Ray(Vector3<float> origin, Vector3<float> direction, Vector3<int> color)
		{
			this->origin = origin;
			this->direction = direction;
			this->color = color;
		}
		__host__ __device__ Ray()
		{
			// Default
		}
	};
	namespace LinAlg
	{
		__host__ __device__ void barycentric(Vector3<float> p, Vector3<float> a, Vector3<float> b, Vector3<float> c, float& u, float& v, float& w)
		{
			Vector3<float> v0 = b - a, v1 = c - a, v2 = p - a;
			float d00 = v0.dot(v0);
			float d01 = v0.dot(v1);
			float d11 = v1.dot(v1);
			float d20 = v2.dot(v0);
			float d21 = v2.dot(v1);
			float denom = d00 * d11 - d01 * d01;
			v = (d11 * d20 - d01 * d21) / denom;
			w = (d00 * d21 - d01 * d20) / denom;
			u = 1.0f - v - w;
		}
		__host__ __device__ Vector3<float> reflect(Vector3<float> v, Vector3<float> normal)
		{
			return v - Vector3<float>(2) * Vector3<float>(normal.dot(v)) * normal;
		}
		__host__ __device__ Vector3<float> rotate(Vector3<float> v, Vector3<float> rotate_by, Vector3<float> center)
		{
			v = v - center;
			float SINF = sinf(rotate_by.z);
			float COSF = cosf(rotate_by.z);
			float output2[1][3] = { { 0, 0, 0 } };
			float input1[1][3];
			float input2[3][3];
			input2[0][0] = COSF;
			input2[0][1] = -SINF;
			input2[0][2] = 0;
			input2[1][0] = SINF;
			input2[1][1] = COSF;
			input2[1][2] = 0;
			input2[2][0] = 0;
			input2[2][1] = 0;
			input2[2][2] = 1;
			input1[0][0] = v.x;
			input1[0][1] = v.y;
			input1[0][2] = v.z;
			for (int _ = 0;_ < 1;_++)
				for (int Y = 0;Y < 3;Y++)
					for (int k = 0;k < 3;k++)
					{
						output2[_][Y] += input1[_][k] * input2[k][Y];
					}
			v = Vector3<float>((float)output2[0][0], (float)output2[0][1], (float)output2[0][2]);
			SINF = sinf(rotate_by.y);
			COSF = cosf(rotate_by.y);
			float output[1][3] = { { 0, 0, 0 } };
			input2[0][0] = COSF;
			input2[0][1] = 0;
			input2[0][2] = SINF;
			input2[1][0] = 0;
			input2[1][1] = 1;
			input2[1][2] = 0;
			input2[2][0] = -SINF;
			input2[2][1] = 0;
			input2[2][2] = COSF;
			input1[0][0] = v.x;
			input1[0][1] = v.y;
			input1[0][2] = v.z;
			for (int _ = 0;_ < 1;_++)
				for (int Y = 0;Y < 3;Y++)
					for (int k = 0;k < 3;k++)
					{
						output[_][Y] += input1[_][k] * input2[k][Y];
					}
			v = Vector3<float>((float)output[0][0], (float)output[0][1], (float)output[0][2]);
			SINF = sinf(rotate_by.x);
			COSF = cosf(rotate_by.x);
			float output4[1][3] = { { 0, 0, 0 } };
			input2[0][0] = 1;
			input2[0][1] = 0;
			input2[0][2] = 0;
			input2[1][0] = 0;
			input2[1][1] = COSF;
			input2[1][2] = -SINF;
			input2[2][0] = 0;
			input2[2][1] = SINF;
			input2[2][2] = COSF;
			input1[0][0] = v.x;
			input1[0][1] = v.y;
			input1[0][2] = v.z;
			for (int _ = 0;_ < 1;_++)
				for (int Y = 0;Y < 3;Y++)
					for (int k = 0;k < 3;k++)
					{
						output4[_][Y] += input1[_][k] * input2[k][Y];
					}
			v = Vector3<float>((float)output4[0][0], (float)output4[0][1], (float)output4[0][2]);

			v = v + center;
			return v;
		}
	}
	class GameObject
	{
	private:
		std::vector<Triangle> triangles;
	public:
		
		std::string name;
		Vector2<Vector3<float>> bounding;
		bool load_from_object_file(std::string sFilename, Vector3<int> color, bool glass = false, bool reflective = false, float reflective_index = 0.0f, int texture_idx = 0, bool textured = false)
		{
			std::ifstream f(sFilename);
			if (!f.is_open())
				return false;

			// Local cache of verts
			std::vector<Vector3<float>> verts;
			int i = -1;
			while (!f.eof())
			{
				char line[128];
				f.getline(line, 128);

				std::strstream s;
				s << line;

				char junk;

				if (line[0] == 'v')
				{
					Vector3<float> v;
					s >> junk >> v.x >> v.y >> v.z;

					verts.push_back(v);
				}

				if (line[0] == 'f')
				{
					i++;
					int f[3];
					s >> junk >> f[0] >> f[1] >> f[2];
					Triangle triangle;
					triangle.set_vertices(
						verts[f[0] - 1],
						verts[f[1] - 1],
						verts[f[2] - 1]
					);
					triangle.color = color;
					triangle.calc_normal();
					triangle.textured = textured;
					triangle.texture_idx = texture_idx;
					triangle.glass = glass;
					triangle.reflective = reflective;
					triangle.reflective_index = reflective_index;
					this->triangles.push_back(triangle);
				}
			}

			return true;
		}
		std::vector<Triangle> get_triangles()
		{
			return this->triangles;
		}
		void translate(Vector3<float> translate_by)
		{
			for (Triangle& triangle : this->triangles)
			{
				triangle.vertices[0] = triangle.vertices[0] + translate_by;
				triangle.vertices[1] = triangle.vertices[1] + translate_by;
				triangle.vertices[2] = triangle.vertices[2] + translate_by;
			}
		}
		void push_back_triangle(Triangle triangle)
		{
			this->triangles.push_back(triangle);
		}
		void rotate(Vector3<float> rotate_by, Vector3<float> center)
		{
			for (Triangle& triangle : this->triangles)
			{


				triangle.vertices[0] = LinAlg::rotate(triangle.vertices[0], rotate_by, center);
				triangle.vertices[1] = LinAlg::rotate(triangle.vertices[1], rotate_by, center);
				triangle.vertices[2] = LinAlg::rotate(triangle.vertices[2], rotate_by, center);
				triangle.calc_normal();
			}
		}
		void calculate_aabb()
		{
			Vector3<float> min_point(1000000000.0f, 1000000000.0f, 1000000000.0f);
			Vector3<float> max_point(-1000000000.0f, -1000000000.0f, -1000000000.0f);
			for (Triangle triangle : this->triangles)
			{
				min_point = Vector3<float>(min(min_point.x, triangle.vertices[0].x), min(min_point.y, triangle.vertices[0].y), min(min_point.z, triangle.vertices[0].z));
				max_point = Vector3<float>(max(max_point.x, triangle.vertices[0].x), max(max_point.y, triangle.vertices[0].y), max(max_point.z, triangle.vertices[0].z));
				min_point = Vector3<float>(min(min_point.x, triangle.vertices[1].x), min(min_point.y, triangle.vertices[1].y), min(min_point.z, triangle.vertices[1].z));
				max_point = Vector3<float>(max(max_point.x, triangle.vertices[1].x), max(max_point.y, triangle.vertices[1].y), max(max_point.z, triangle.vertices[1].z));
				min_point = Vector3<float>(min(min_point.x, triangle.vertices[2].x), min(min_point.y, triangle.vertices[2].y), min(min_point.z, triangle.vertices[2].z));
				max_point = Vector3<float>(max(max_point.x, triangle.vertices[2].x), max(max_point.y, triangle.vertices[2].y), max(max_point.z, triangle.vertices[2].z));
			}
			this->bounding.x = min_point;
			this->bounding.y = max_point;
		}
	};
	bool load_multiple_from_object_file(std::vector<GameObject> &output_game_objects, std::string sFilename, Vector3<int> color, bool glass = false, bool reflective = false, float reflective_index = 0.0f, int texture_idx = 0, bool textured = false)
	{
		std::ifstream f(sFilename);
		if (!f.is_open())
			return false;

		// Local cache of verts
		std::vector<Vector3<float>> verts;
		int i = -1;
		
		GameObject cur_game_object;
		bool passed;
		while (!f.eof())
		{
			char line[128];
			f.getline(line, 128);

			std::strstream s;
			s << line;

			char junk;
			if (line[0] == 'o')
			{
				if (passed)
				{
					cur_game_object.calculate_aabb();
					output_game_objects.push_back(cur_game_object);
				}
				GameObject new_object;
				
				s >> junk >> new_object.name;
				cur_game_object = new_object;

				passed = true;
			}
			if (line[0] == 'v')
			{
				Vector3<float> v;
				s >> junk >> v.x >> v.y >> v.z;

				verts.push_back(v);
			}

			if (line[0] == 'f')
			{
				i++;
				int f[3];
				s >> junk >> f[0] >> f[1] >> f[2];
				Triangle triangle;
				triangle.set_vertices(
					verts[f[0] - 1],
					verts[f[1] - 1],
					verts[f[2] - 1]
				);
				triangle.color = color;
				triangle.calc_normal();
				triangle.textured = textured;
				triangle.texture_idx = texture_idx;
				triangle.glass = glass;
				triangle.reflective = reflective;
				triangle.reflective_index = reflective_index;
				cur_game_object.push_back_triangle(triangle);
			}
		}

		return true;
	}
	namespace Intersections
	{
		__host__ __device__ bool ray_triangle(Triangle triangle, float& t, Ray& ray)
		{
			Vector3<float> v0v1 = triangle.vertices[1] - triangle.vertices[0];
			Vector3<float> v0v2 = triangle.vertices[2] - triangle.vertices[0];
			Vector3<float> pvec = ray.direction.cross(v0v2);
			float det = v0v1.dot(pvec);
			// if (fabs(det) < 0.00001f) return false;
			float invDet = 1.0f / det;

			Vector3<float> tvec = ray.origin - triangle.vertices[0];
			float u = tvec.dot(pvec) * invDet;
			if (u < 0 || u > 1) return false;

			Vector3<float> qvec = tvec.cross(v0v1);
			float v = ray.direction.dot(qvec) * invDet;
			if (v < 0 || u + v > 1) return false;


			t = v0v2.dot(qvec) * invDet;
			if (t < 0) return false;

			ray.u = u;
			ray.v = v;
			return true;
		}
		__host__ __device__ bool ray_aabb(Ray ray, float& t, Vector2<Vector3<float>> minmax)
		{
			if (ray.origin.x > minmax.x.x && ray.origin.x < minmax.y.x && ray.origin.y > minmax.x.y && ray.origin.y < minmax.y.y && ray.origin.z > minmax.x.z && ray.origin.z < minmax.y.z)
			{
				t = 0.0f;
				return true;
			}
			Vector3<float> tMin = (minmax.x - ray.origin) / ray.direction;
			Vector3<float> tMax = (minmax.y - ray.origin) / ray.direction;
			Vector3<float> t1 = Vector3<float>(min(tMin.x, tMax.x), min(tMin.y, tMax.y), min(tMin.z, tMax.z));
			Vector3<float> t2 = Vector3<float>(max(tMin.x, tMax.x), max(tMin.y, tMax.y), max(tMin.z, tMax.z));
			float tNear = max(max(t1.x, t1.y), t1.z);
			float tFar = min(min(t2.x, t2.y), t2.z);
			if (tNear - 1 > tFar) return false;
			if (tNear < 0) return false;
			t = tNear;
			return true;
		}
		__host__ __device__ bool ray_sphere(Vector3<float> center, float radius, float& t, Ray ray)
		{
			Vector3<float> oc = ray.origin - center;
			float a = ray.direction.dot(ray.direction);
			float B = 2.0 * oc.dot(ray.direction);
			float c = oc.dot(oc) - radius * radius;
			float discriminant = B * B - 4 * a * c;
			if (discriminant >= 0.0) {
				float numerator = -B - sqrtf(discriminant);
				if (numerator > 0.0) {
					float dist = numerator / (2.0 * a);
					t = dist;
					return true;
				}
			}
			return false;
		}
		bool Raycast(float& t, Ray ray, std::vector<Triangle> triangles)
		{
			float curT = t;
			for (Triangle triangle : triangles)
			{
				float temp;
				if (ray_triangle(triangle, temp, ray))
				{
					if (temp < t)
					{
						t = temp;
					}
				}
			}
			if (t < curT)
			{
				return true;
			}
			return false;
		}
		__device__ bool device_raycast(Triangle* triangles, Ray ray, Triangle& out_tri, float &out_t, float* boundings, int bounding_count)
		{
			
			int excluded[100];
			int excluded_count = 0;
			
			while (excluded_count != bounding_count)
			{
				float t2 = 10000.0f;
				int start_tri = -1;
				int end_tri = -1;
				int b_idx2 = 0;
				for (int b_idx = 0;b_idx < bounding_count;b_idx++)
				{
					Vector3<float> min_point(boundings[b_idx * 8], boundings[b_idx * 8 + 1], boundings[b_idx * 8 + 2]);
					Vector3<float> max_point(boundings[b_idx * 8 + 3], boundings[b_idx * 8 + 4], boundings[b_idx * 8 + 5]);
					float temp;
					if (ray_aabb(ray, temp, Vector2<Vector3<float>>(min_point, max_point)))
					{

						if (temp < t2)
						{
							bool should_continue = false;
							for (int j = 0;j < excluded_count;j++)
							{
								if (b_idx == excluded[j]) should_continue = true;
							}
							if (should_continue) continue;
							start_tri = (int)boundings[b_idx * 8 + 6];
							end_tri = (int)boundings[b_idx * 8 + 7];
							t2 = temp;
							b_idx2 = b_idx;
						}
					}
				}


				if (start_tri == -1)
				{
					return false;
				}
				float t = 10000.0f;
				Triangle triangle;
				for (int t_idx = start_tri;t_idx < end_tri + 1;t_idx++)
				{
					Triangle triangle2 = triangles[t_idx];
					float dist;
					if (ray_triangle(triangle2, dist, ray))
					{
						if (dist < t)
						{
							t = dist;
							triangle = triangle2;
						}
					}
				}
				if (t == 10000.0f)
				{
					excluded[excluded_count] = b_idx2;
					excluded_count++;
				}
				else
				{
					out_tri = triangle;
					out_t = t;
					return true;
				}
			}
		}
	}
	class Light
	{
	public:
		Vector3<float> pos;
		Vector3<int> color;
		float intensity;
		__host__ __device__ Light(Vector3<float> pos, Vector3<int> color, float intensity)
		{
			this->pos = pos;
			this->color = color;
			this->intensity = intensity;
		}
		__host__ __device__ Light()
		{
			// Default
		}
	};
	
	class Camera
	{
	public:
		Vector3<float> pos;
		Vector2<float> rot;
		Camera(Vector3<float> pos, Vector2<float> rot)
		{
			this->pos = pos;
			this->rot = rot;
		}
		Camera()
		{
			// Default
		}
	};
	namespace Wrappers
	{
		class render_result_wrapper
		{
		public:
			int* r;
			int* g;
			int* b;
		};
	
	}
	
	
	namespace Rendering
	{
		__global__ void kernel_render_screen(Triangle* triangles, int triangles_count, float* boundings, int bounding_count, Camera camera, Light* lights, int light_count, Vector3<int> screen_background, Vector2<int> screen_resolution, Texture2D* textures, int* out_r, int* out_g, int* out_b)
		{
			float i = (float)threadIdx.x;
			float j = (float)blockIdx.x;
			Ray ray;
			ray.origin = camera.pos;
			ray.direction = Vector3<float>(0.5f, -(i / screen_resolution.x - 0.5f), (j / screen_resolution.y - 0.5f)).normalize();
			float SINF = sinf(camera.rot.x);
			float COSF = cosf(camera.rot.x);
			float output2[1][3] = { { 0, 0, 0 } };
			float input1[1][3];
			float input2[3][3];
			input2[0][0] = COSF;
			input2[0][1] = -SINF;
			input2[0][2] = 0;
		    input2[1][0] = SINF;
			input2[1][1] = COSF;
			input2[1][2] = 0;
			input2[2][0] = 0;
			input2[2][1] = 0;
			input2[2][2] = 1;
			input1[0][0] = ray.direction.x;
			input1[0][1] = ray.direction.y;
			input1[0][2] = ray.direction.z;
			for (int _ = 0;_ < 1;_++)
				for (int Y = 0;Y < 3;Y++)
					for (int k = 0;k < 3;k++)
					{
						output2[_][Y] += input1[_][k] * input2[k][Y];
					}
			ray.direction = Vector3<float>((float)output2[0][0], (float)output2[0][1], (float)output2[0][2]);
			SINF = sinf(camera.rot.y);
			COSF = cosf(camera.rot.y);
			float output23[1][3] = { { 0, 0, 0 } };
			input2[0][0] = COSF;
			input2[0][1] = 0;
			input2[0][2] = SINF;
			input2[1][0] = 0;
			input2[2][0] = -SINF;
			input2[1][1] = 1;
			input2[1][2] = 0;
			input2[2][1] = 0;
			input2[2][2] = COSF;
			input1[0][0] = ray.direction.x;
			input1[0][1] = ray.direction.y;
			input1[0][2] = ray.direction.z;
			for (int _ = 0;_ < 1;_++)
				for (int Y = 0;Y < 3;Y++)
					for (int k = 0;k < 3;k++)
					{
						output23[_][Y] += input1[_][k] * input2[k][Y];
					}
			ray.direction = Vector3<float>((float)output23[0][0], (float)output23[0][1], (float)output23[0][2]);

			ray.color = screen_background;
			bool reflected = false;
			bool glass = false;
			for (int depth_idx = 0;depth_idx < 3;depth_idx++)
			{
				float t = 10000.0f;
				Triangle out_tri;
				
				if (Intersections::device_raycast(triangles, ray, out_tri, t, boundings, bounding_count))
				{
					Vector3<float> intersection = ray.origin + Vector3<float>(t) * ray.direction;
					float u, v, w;
					
					LinAlg::barycentric(intersection, out_tri.vertices[0], out_tri.vertices[1], out_tri.vertices[2], u, v, w);
					if (out_tri.normal.dot((ray.origin - intersection).normalize()) > 0.0f) out_tri.normal = out_tri.normal * Vector3<float>(-1.0f);
					if (out_tri.textured && textures[out_tri.texture_idx].normal)
					{
						float tex_color[3] = { textures[out_tri.texture_idx].normal_data[(int)(u * 256)][(int)(w * 128)][0], textures[out_tri.texture_idx].normal_data[(int)(u * 256)][(int)(w * 128)][1], textures[out_tri.texture_idx].normal_data[(int)(u * 256)][(int)(w * 128)][2] };
						Vector3<float> tex_vector(tex_color[0], tex_color[1], tex_color[2]);
						float d = fast_inverse_sqrt(out_tri.normal.x * out_tri.normal.x + out_tri.normal.y * out_tri.normal.y),
							float d2 = fast_inverse_sqrt(tex_vector.x * tex_vector.x + tex_vector.y * tex_vector.y);
						float phi = atanf(out_tri.normal.y / out_tri.normal.x),
							theta = out_tri.normal.z;
						float phi2 = atanf(tex_vector.y / tex_vector.x),
							theta2 = tex_vector.z;
						phi2 += phi;
						theta2 += theta;
						d2 += d;
						out_tri.normal = Vector3<float>(
							d2 * cosf(phi2),
							d2 * sinf(phi2),
							tex_vector.z + out_tri.normal.z
							
							);
					}
					if (out_tri.normal.dot((ray.origin - intersection).normalize()) > 0.0f) out_tri.normal = out_tri.normal * Vector3<float>(-1.0f);
					out_tri.normal = out_tri.normal.normalize();
					float avg_lums = 0.0f;
					Vector3<float> avg_col;
					for (int light_idx = 0;light_idx < light_count;light_idx++)
					{
						Vector3<float> itol = (lights[light_idx].pos - intersection);
						float itol_mag = itol.magnitude();
						float intensity = max(-itol.normalize().dot(out_tri.normal), 0.0f);
						intensity *= lights[light_idx].intensity / itol_mag;
						avg_col = avg_col + Vector3<float>(lights[light_idx].color.x * intensity, lights[light_idx].color.y * intensity, lights[light_idx].color.z * intensity);
						Ray ray_to_sun(intersection, (lights[light_idx].pos - intersection).normalize(), ray.color);
						ray_to_sun.origin = ray_to_sun.origin + ray_to_sun.direction * Vector3<float>(0.001f);
						int shadowHit = 1;
						float t2 = itol_mag;
						float temp;
						Triangle tri_temp;
						if (Intersections::device_raycast(triangles, ray_to_sun, tri_temp, temp, boundings, bounding_count))
						{
							if (temp < t2) shadowHit = 0;
						}
						avg_lums += intensity * shadowHit;


					}
					avg_col = avg_col.normalize();
					
					if (out_tri.textured && textures[out_tri.texture_idx].albedo)
					{
						
						
						float tex_color[3] = { textures[out_tri.texture_idx].data[(int)(u * 512)][(int)(w * 512)][0], textures[out_tri.texture_idx].data[(int)(u * 512)][(int)(w * 512)][1], textures[out_tri.texture_idx].data[(int)(u * 512)][(int)(w * 512)][2] };
						out_tri.color = Vector3<int>(tex_color[0], tex_color[1], tex_color[2]);
					}
					Vector3<int> this_col = Vector3<int>(min(out_tri.color.x * avg_col.x * avg_lums, 255.0f), min(out_tri.color.y * avg_col.y * avg_lums, 255.0f), min(out_tri.color.z * avg_col.z * avg_lums, 255.0f));
					
					if (out_tri.reflective || reflected)
					{
						reflected = true;
						ray.direction = LinAlg::reflect(ray.direction, out_tri.normal); // + Vector3<float>(curand_uniform(&rand_states[tid + depth_idx]) * 0.1f - 0.05f, curand_uniform(&rand_states[tid + depth_idx + 1]) * 0.1f - 0.05f, curand_uniform(&rand_states[tid + depth_idx + 2]) * 0.1f - 0.05f);
						ray.origin = intersection + ray.direction * Vector3<float>(0.01f);
						if (depth_idx == 0) ray.color = this_col;
						else
						{
							ray.color = (Vector3<int>(this_col.x * (1.0f - out_tri.reflective_index), this_col.y * (1.0f - out_tri.reflective_index), this_col.z * (1.0f - out_tri.reflective_index)) + Vector3<int>(ray.color.x * out_tri.reflective_index, ray.color.y * out_tri.reflective_index, ray.color.z * out_tri.reflective_index));
						}
					}
					else if (out_tri.glass || glass)
					{
						glass = true;
						ray.origin = intersection + ray.direction * Vector3<float>(0.01f);
						if (depth_idx == 0) ray.color = this_col;
						else ray.color = (Vector3<int>(this_col.x * 0.95f, this_col.y * 0.95f, this_col.z * 0.95f) + Vector3<int>(ray.color.x * 0.05f, ray.color.y * 0.05f, ray.color.z * 0.05f));
					}
					else
					{
						ray.color = this_col;
						break;
					}
				}
				else
				{
					ray.color = (Vector3<int>(ray.color.x * 0.875f, ray.color.y * 0.875f, ray.color.z * 0.875f) + Vector3<int>(screen_background.x * 0.125f, screen_background.y * 0.125f, screen_background.z * 0.125f));
				}
			}
			out_r[(int)j + (int)i * screen_resolution.y] = ray.color.x;
			out_g[(int)j + (int)i * screen_resolution.y] = ray.color.y;
			out_b[(int)j + (int)i * screen_resolution.y] = ray.color.z;

		}
		int data_r[500 * 500];
		int data_g[500 * 500];
		int data_b[500 * 500];
		Wrappers::render_result_wrapper render_screen(Triangle* triangles, int triangle_count, float* boundings, int bounding_count, Vector2<int> screen_resolution, Vector3<int> screen_background, int ray_depth, Camera camera, Light* lights, int light_count, Texture2D* textures, int texture_count)
		{
			Triangle* gpu_triangles = nullptr;
			cudaMalloc(&gpu_triangles, sizeof(Triangle) * triangle_count);
			cudaMemcpy(gpu_triangles, triangles, sizeof(Triangle) * triangle_count, cudaMemcpyHostToDevice);
			int* dev_outputR = nullptr;
			
			cudaMalloc(&dev_outputR, (int)screen_resolution.x * screen_resolution.y * sizeof(int));
			cudaMemcpy(dev_outputR, data_r, (int)screen_resolution.x * screen_resolution.y * sizeof(int), cudaMemcpyHostToDevice);
			int* dev_outputG = nullptr;
			
			cudaMalloc(&dev_outputG, (int)screen_resolution.x * screen_resolution.y * sizeof(int));
			cudaMemcpy(dev_outputG, data_g, (int)screen_resolution.x * screen_resolution.y * sizeof(int), cudaMemcpyHostToDevice);
			int* dev_outputB = nullptr;
			
			cudaMalloc(&dev_outputB, (int)screen_resolution.x * screen_resolution.y * sizeof(int));
			cudaMemcpy(dev_outputB, data_b, (int)screen_resolution.x * screen_resolution.y * sizeof(int), cudaMemcpyHostToDevice);
			Light* dev_lights = nullptr;
			cudaMalloc(&dev_lights, sizeof(Light) * light_count);
			cudaMemcpy(dev_lights, lights, sizeof(Light) * light_count, cudaMemcpyHostToDevice);
			Texture2D* dev_textures = nullptr;
			cudaMalloc(&dev_textures, sizeof(Texture2D) * texture_count);
			cudaMemcpy(dev_textures, textures, sizeof(Texture2D) * texture_count, cudaMemcpyHostToDevice);

			float* dev_boundings;
			cudaMalloc(&dev_boundings, 8 * sizeof(float) * bounding_count);
			cudaMemcpy(dev_boundings, boundings, 8 * sizeof(float) * bounding_count, cudaMemcpyHostToDevice);

			float t = clock();
			kernel_render_screen<<<(int)screen_resolution.y, (int)screen_resolution.x>>>(gpu_triangles, triangle_count, dev_boundings, bounding_count, camera, dev_lights, light_count, screen_background, Vector2<int>(screen_resolution.x, screen_resolution.y), dev_textures, dev_outputR, dev_outputG, dev_outputB);
			cudaDeviceSynchronize();
			std::cout << "RENDER FPS: " << (clock() - t) / CLOCKS_PER_SEC << " ERROR: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
			Wrappers::render_result_wrapper wrapper;
			cudaMemcpy(data_r, dev_outputR, (int)screen_resolution.x * screen_resolution.y * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(data_g, dev_outputG, (int)screen_resolution.x * screen_resolution.y * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(data_b, dev_outputB, (int)screen_resolution.x * screen_resolution.y * sizeof(int), cudaMemcpyDeviceToHost);
			wrapper.r = data_r;
			wrapper.g = data_g;
			wrapper.b = data_b;
			cudaFree(dev_outputR);
			cudaFree(dev_outputG);
			cudaFree(dev_outputB);
			cudaFree(gpu_triangles);
			cudaFree(dev_lights);
			cudaFree(dev_textures);
			cudaFree(dev_boundings);
			return wrapper;
		}
	}
	class Scene
	{
	private:
		std::vector<GameObject> _scene;
		std::vector<Light> lights;
		Vector2<float> inv_resolution;
		std::vector<float> boundary_data;
	public:
		cv::Mat canvas;
		Vector2<int> stretch_to;
		Camera camera;
		
		Vector3<int> color_background;
		std::vector<Texture2D> textures;
		bool bloom = false;
		Scene(int CANVAS_TYPE, Vector2<int> resolution, Vector2<int> stretch_resolution_to, Camera camera, Vector3<int> color_background)
		{
			this->canvas = cv::Mat::zeros(cv::Size(resolution.x, resolution.y), CANVAS_TYPE);
			this->camera = camera;
			this->color_background = color_background;
			this->stretch_to = stretch_resolution_to;
		}
		void add_game_object(GameObject game_object)
		{
			this->_scene.push_back(game_object);
		}
		std::vector<GameObject> get_game_objects()
		{
			return this->_scene;
		}
		void set_game_object(int idx, GameObject new_object)
		{
			this->_scene[idx] = new_object;
		}
		void add_light(Light light)
		{
			this->lights.push_back(light);
		}
		std::vector<Light> get_lights()
		{
			return this->lights;
		}
		void set_light(const int idx, const Light new_light)
		{
			this->lights[idx] = new_light;
		}
		bool load_lights_from_object_file(std::string sFilename, Vector3<int> color, float intensity)
		{
			std::ifstream f(sFilename);
			if (!f.is_open())
				return false;
			while (!f.eof())
			{
				char line[128];
				f.getline(line, 128);

				std::strstream s;
				s << line;

				char junk;

				if (line[0] == 'v')
				{
					Vector3<float> v;
					s >> junk >> v.x >> v.y >> v.z;
					this->lights.push_back(Light(v, color, intensity));
				}
			}

			return true;
		}
		void load_gameobjects_from_vector(std::vector<GameObject> game_objects)
		{
			for (GameObject game_object : game_objects)
			{
				this->add_game_object(game_object);
			}
		}
		int find_gameobject_by_name(std::string name)
		{
			for (int i = 0;i < this->_scene.size();i++)
			{
				if (this->_scene[i].name == name) return i;
			}
			return -1;
		}
		void bake_boundary_data()
		{
			std::vector<float> bounding_data;
			int triangle_counter = 0;
			for (GameObject& game_object : this->_scene)
			{
				std::vector<Triangle> game_object_triangles = game_object.get_triangles();
				bounding_data.push_back(game_object.bounding.x.x);
				bounding_data.push_back(game_object.bounding.x.y);
				bounding_data.push_back(game_object.bounding.x.z);
				bounding_data.push_back(game_object.bounding.y.x);
				bounding_data.push_back(game_object.bounding.y.y);
				bounding_data.push_back(game_object.bounding.y.z);
				bounding_data.push_back((float)triangle_counter);
				bounding_data.push_back((float)triangle_counter + game_object_triangles.size() - 1);
				triangle_counter += game_object_triangles.size();

			}
			this->boundary_data = bounding_data;
		}
		void render()
		{
			std::vector<Triangle> triangles;
			for (GameObject& game_object : this->_scene)
			{
				std::vector<Triangle> game_object_triangles = game_object.get_triangles();
				triangles.insert(triangles.end(), game_object_triangles.begin(), game_object_triangles.end());

			}
			Wrappers::render_result_wrapper col = Rendering::render_screen(triangles.data(), triangles.size(), this->boundary_data.data(), this->_scene.size(), Vector2<int>(this->canvas.rows, this->canvas.cols), this->color_background, 1, this->camera, &this->lights[0], this->lights.size(), &this->textures[0], this->textures.size());
			for (int y = 0;y < this->canvas.rows;y++)
			{
				for (int x = 0;x < this->canvas.cols;x++)
				{

					cv::Vec3b& at = this->canvas.at<cv::Vec3b>(y, x);
					at.val[0] = col.r[(x + y * this->canvas.cols)];
					at.val[1] = col.g[(x + y * this->canvas.cols)];
					at.val[2] = col.b[(x + y * this->canvas.cols)];
				}
			}
			if (this->bloom)
			{
				cv::Mat gray;
				cv::cvtColor(this->canvas, gray, cv::COLOR_BGR2GRAY);
				cv::threshold(gray, gray, 200, 255, cv::THRESH_BINARY);
				cv::GaussianBlur(gray, gray, cv::Size(0, 0), 10);
				cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
				cv::addWeighted(this->canvas, 1, gray, 6, 0, this->canvas);
			}
			cv::GaussianBlur(this->canvas, this->canvas, cv::Size(0, 0), 1.0);
			cv::Mat output_canvas;
			cv::resize(this->canvas, output_canvas, cv::Size(this->stretch_to.x, this->stretch_to.y));
			cv::imshow("Krov Engine", output_canvas);
			cv::waitKey(1);
		}
	};
	namespace GenTypes
	{
		template <class T>
		__host__ __device__ T FILL_ZEROS(int idx)
		{
			return (T)0;
		}
		template <class T>
		__host__ __device__ T FILL_ONES(int idx)
		{
			return (T)1;
		}
		template <class T>
		__host__ __device__ T FILL_IDX(int idx)
		{
			return (T)idx;
		}
	}
}