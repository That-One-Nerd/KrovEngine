#undef _SCRT_STARTUP_MAIN
#include <iostream>	

#include "opencv2/opencv.hpp"

#include "glm/glm.hpp"
#include "glm/ext.hpp"
#include <chrono>
#include <fstream>
#include <strstream>
#include <math.h>
#include <Windows.h>
#include "lua.hpp"
#include <vector>
#include <algorithm>
#include <strstream>
#include <filesystem>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cudamathOld.cuh"
#include <thrust/device_vector.h>
#include "OpenImageDenoise/oidn.hpp"
#define RESOLUTION 512
#define ALLOC_MEM_TRIS_NUM 800
__host__ __device__ struct Triangle
{
public:
	float points[13] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	float3 col = { 0, 1, 0 };
	float3 normal = { 0, 0, 0 };
};
void calculateBounding(Triangle& triangle)
{
	float* points = triangle.points;
	triangle.points[9] = min(points[0], min(points[3], points[6]));
	triangle.points[10] = max(points[0], max(points[3], points[6]));
	triangle.points[11] = min(points[1], min(points[4], points[7]));
	triangle.points[12] = max(points[1], max(points[4], points[7]));
}
void calculateNormal(Triangle& triangle)
{
	float3 normal = normalize(cross(sub({ triangle.points[6], triangle.points[7], triangle.points[8] }, { triangle.points[0], triangle.points[1], triangle.points[2] }), sub({ triangle.points[3], triangle.points[4], triangle.points[5] }, { triangle.points[0], triangle.points[1], triangle.points[2] })));
	triangle.normal = normal;
}
__host__ __device__ struct Mesh
{
public:
	glm::vec3 position;
	std::vector<Triangle> triangles;
};
__host__ __device__ bool rayTriangleIntersect(float3 v0, float3 v1, float3 v2, float& t, float3 rayPos, float3 rayVec)
{
	float3 dir = rayVec;
	float3 orig = rayPos;
	float3 v0v1 = sub(v1, v0);
	float3 v0v2 = sub(v2, v0);
	float3 N = cross(v0v1, v0v2);
	float area2 = length(N);
	float NdotRayDirection = dot(N, dir);
	if (fabs(NdotRayDirection) < 0.000001f)
		return false;
	float d = -dot(N, v0);
	t = -(dot(N, orig) + d) / NdotRayDirection;
	if (t < 0) return false;
	float3 P = add(orig, mult({ t, t, t }, dir));
	float3 C;
	float3 edge0 = sub(v1, v0);
	float3 vp0 = sub(P, v0);
	C = cross(edge0, vp0);
	if (dot(N, C) < 0) return false;
	float3 edge1 = sub(v2, v1);
	float3 vp1 = sub(P, v1);
	C = cross(edge1, vp1);
	if (dot(N, C) < 0)  return false;
	float3 edge2 = sub(v0, v2);
	float3 vp2 = sub(P, v2);
	C = cross(edge2, vp2);
	if (dot(N, C) < 0) return false;
	return true;
}
__global__ void _draw_pix(int y, float camX, float camY, float camZ, float sunX, float sunY, float sunZ, float playerX, float playerY, float playerZ, float degreesXZ, float degreesYZ, curandState* rand_state, Triangle* tris, int depth, int* r, int* g, int* b)
{
	float i = threadIdx.x;
	float j = blockIdx.x;
	float3 rayVec = normalize({ -(i / RESOLUTION - 0.5f), -((j + y * RESOLUTION * 0.25f) / RESOLUTION - 0.5f), 1.0f });
	float3 rayPos{ camX, camY, camZ };
	int3 fragColor{ 255, 100, 100 };
	float SINF = sinf(degreesXZ);
	float COSF = cosf(degreesXZ);
	float output2[1][3] = { { 0, 0, 0 } };
	float input1[1][3];
	float input2[3][3];
	input2[0][0] = 1;
	input2[0][1] = 0;
	input2[0][2] = 0;
	input2[1][0] = 0;
	input2[2][0] = 0;
	input2[1][1] = COSF;
	input2[1][2] = -SINF;
	input2[2][1] = SINF;
	input2[2][2] = COSF;
	input1[0][0] = rayVec.x;
	input1[0][1] = rayVec.y;
	input1[0][2] = rayVec.z;
	for (int _ = 0;_ < 1;_++)
		for (int Y = 0;Y < 3;Y++)
			for (int k = 0;k < 3;k++)
			{
				output2[_][Y] += input1[_][k] * input2[k][Y];
			}
	rayVec = { (float)output2[0][0], (float)output2[0][1], (float)output2[0][2] };
	SINF = sinf(degreesYZ);
	COSF = cosf(degreesYZ);
	float output22[1][3] = { { 0, 0, 0 } };
	input2[0][0] = COSF;
	input2[0][1] = 0;
	input2[0][2] = SINF;
	input2[1][0] = 0;
	input2[2][0] = -SINF;
	input2[1][1] = 1;
	input2[1][2] = 0;
	input2[2][1] = 0;
	input2[2][2] = COSF;
	input1[0][0] = rayVec.x;
	input1[0][1] = rayVec.y;
	input1[0][2] = rayVec.z;
	for (int _ = 0;_ < 1;_++)
		for (int Y = 0;Y < 3;Y++)
			for (int k = 0;k < 3;k++)
			{
				output22[_][Y] += input1[_][k] * input2[k][Y];
			}
	rayVec = { (float)output22[0][0], (float)output22[0][1], (float)output22[0][2] };
	rayVec = normalize(rayVec);
	Triangle closest;
	float closeT = 1000.0f;
	for (int i = 0;i < ALLOC_MEM_TRIS_NUM;i++)
	{
		float3 p1{ tris[i].points[0], tris[i].points[1], tris[i].points[2] };
		float3 p2{ tris[i].points[3], tris[i].points[4], tris[i].points[5] };
		float3 p3{ tris[i].points[6], tris[i].points[7], tris[i].points[8] };
		float t;
		if (rayTriangleIntersect(p1, p2, p3, t, rayPos, rayVec))
		{
			if (t < closeT)
			{
				closeT = t;
				closest = tris[i];
			}
		}
	}
	bool playerHit = false;
	float3 oc = sub(rayPos, { playerX, playerY, playerZ });
	float a = dot(rayVec, rayVec);
	float B = 2.0 * dot(oc, rayVec);
	float c = dot(oc, oc) - 0.2f * 0.2f;
	float discriminant = B * B - 4 * a * c;
	if (discriminant >= 0.0) {
		float numerator = -B - sqrtf(discriminant);
		if (numerator > 0.0) {
			float dist = numerator / (2.0 * a);
			if (dist < closeT)
			{
				playerHit = true;
				float3 intersect = add(rayPos, mult({ dist, dist, dist }, rayVec));
				float3 normal = normalize(sub({ playerX, playerY, playerZ }, intersect));
				float lums = max(dot(normal, normalize(sub(intersect, { sunX, sunY, sunZ }))), 0.0f) * 255;
				fragColor = { (int)(lums * 0), (int)(lums * 1), (int)(lums * 0) };
			}
		}
	}
	if (closeT < 1000.0f && !playerHit)
	{
		float3 intersect = add(rayPos, mult({ closeT, closeT, closeT }, rayVec));
		float totalR = 0, totalG = 0, totalB = 0;
		for (int n = 0;n < depth;n++)
		{
			rayVec = normalize(sub(normalize(sub({ sunX, sunY, sunZ }, intersect)), { curand_uniform(rand_state + n + threadIdx.x) * 0.5f - 0.25f, curand_uniform(rand_state + n + 1 + blockIdx.x) * 0.5f - 0.25f, curand_uniform(rand_state + 2 + n + threadIdx.x + blockIdx.x) * 0.5f - 0.25f }));
			rayPos = add(intersect, mult(rayVec, { 0.005f, 0.005f, 0.005f }));
			closeT = 1000.0f;
			Triangle closest2;
			for (int i = 0;i < ALLOC_MEM_TRIS_NUM;i++)
			{
				float3 p1{ tris[i].points[0], tris[i].points[1], tris[i].points[2] };
				float3 p2{ tris[i].points[3], tris[i].points[4], tris[i].points[5] };
				float3 p3{ tris[i].points[6], tris[i].points[7], tris[i].points[8] };
				float t;
				if (rayTriangleIntersect(p1, p2, p3, t, rayPos, rayVec))
				{
					if (t < closeT)
					{
						closeT = t;
						closest2 = tris[i];
					}
				}
			}
			float3 oc = sub(rayPos, { playerX, playerY, playerZ });
			float a = dot(rayVec, rayVec);
			float B = 2.0 * dot(oc, rayVec);
			float c = dot(oc, oc) - 0.2f * 0.2f;
			float discriminant = B * B - 4 * a * c;
			if (discriminant >= 0.0) {
				float numerator = -B - sqrtf(discriminant);
				if (numerator > 0.0) {
					float dist = numerator / (2.0 * a);
					if (dist < closeT)
					{
						closeT = dist;
					}
				}
			}
			float lums = 0;
			if (closeT < 1000.0f)
			{
				lums = max(dot(closest.normal, normalize(sub(intersect, { sunX, sunY, sunZ }))), 0.0f) * 40;
			}
			else
			{
				lums = max(dot(closest.normal, normalize(sub(intersect, { sunX, sunY, sunZ }))), 0.0f) * 255;
			}
			totalR += (lums * closest.col.x);
			totalG += (lums * closest.col.y);
			totalB += (lums * closest.col.z);
		}
		totalR /= depth;
		totalG /= depth;
		totalB /= depth;
		fragColor = { (int)totalR, (int)totalG, (int)totalB };
	}

	r[(int)i + (int)j * RESOLUTION] = fragColor.x;
	g[(int)i + (int)j * RESOLUTION] = fragColor.y;
	b[(int)i + (int)j * RESOLUTION] = fragColor.z;
}
class Wrapper
{
public:
	int* r;
	int* g;
	int* b;
};
bool loadFromObjectFile(std::string sFilename, std::vector<Triangle>& anyData)
{
	std::ifstream f(sFilename);
	if (!f.is_open())
		return false;

	// Local cache of verts
	std::vector<glm::vec3> verts;
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
			glm::vec3 v;
			s >> junk >> v[0] >> v[1] >> v[2];
			verts.push_back(v);
		}

		if (line[0] == 'f')
		{
			i++;
			int f[3];
			s >> junk >> f[0] >> f[1] >> f[2];
			if (i < ALLOC_MEM_TRIS_NUM)
			{
				anyData[i].points[0] = verts[f[0] - 1][0];
				anyData[i].points[1] = verts[f[0] - 1][1];
				anyData[i].points[2] = verts[f[0] - 1][2] + 10.0f;
				anyData[i].points[3] = verts[f[1] - 1][0];
				anyData[i].points[4] = verts[f[1] - 1][1];
				anyData[i].points[5] = verts[f[1] - 1][2] + 10.0f;
				anyData[i].points[6] = verts[f[2] - 1][0];
				anyData[i].points[7] = verts[f[2] - 1][1];
				anyData[i].points[8] = verts[f[2] - 1][2] + 10.0f;
				anyData[i].col = { (float)(rand() % 255) / 255, (float)(rand() % 255) / 255, (float)(rand() % 255) / 255 };
				calculateBounding(anyData[i]);
				calculateNormal(anyData[i]);
			}
		}
	}

	return true;
}
Wrapper helper(int y, float camX, float camY, float camZ, float sunX, float sunY, float sunZ, float playerX, float playerY, float playerZ, float degreesXZ, float degreesYZ, Triangle* tris, curandState* state, int depth)
{
	Triangle* dev_tris = nullptr;
	cudaMalloc(&dev_tris, sizeof(Triangle) * ALLOC_MEM_TRIS_NUM);
	cudaMemcpy(dev_tris, tris, sizeof(Triangle) * ALLOC_MEM_TRIS_NUM, cudaMemcpyHostToDevice);
	int* dev_outputR = nullptr;
	int outputR[(RESOLUTION * (RESOLUTION / 4))] = { 255 };
	cudaMalloc(&dev_outputR, (RESOLUTION * (RESOLUTION / 4)) * sizeof(int));
	cudaMemcpy(dev_outputR, outputR, (RESOLUTION * (RESOLUTION / 4)) * sizeof(int), cudaMemcpyHostToDevice);
	int* dev_outputG = nullptr;
	int outputG[(RESOLUTION * (RESOLUTION / 4))] = { 255 };
	cudaMalloc(&dev_outputG, (RESOLUTION * (RESOLUTION / 4)) * sizeof(int));
	cudaMemcpy(dev_outputG, outputG, (RESOLUTION * (RESOLUTION / 4)) * sizeof(int), cudaMemcpyHostToDevice);
	int* dev_outputB = nullptr;
	int outputB[(RESOLUTION * (RESOLUTION / 4))] = { 255 };
	cudaMalloc(&dev_outputB, (RESOLUTION * (RESOLUTION / 4)) * sizeof(int));
	cudaMemcpy(dev_outputB, outputB, (RESOLUTION * (RESOLUTION / 4)) * sizeof(int), cudaMemcpyHostToDevice);
	curandState* dev_state = nullptr;
	cudaMalloc(&dev_state, sizeof(curandState));
	cudaMemcpy(dev_state, state, sizeof(curandState), cudaMemcpyHostToDevice);
	_draw_pix << <RESOLUTION / 4, RESOLUTION >> > (y, camX, camY, camZ, sunX, sunY, sunZ, playerX, playerY, playerZ, degreesXZ, degreesYZ, dev_state, dev_tris, depth, dev_outputR, dev_outputG, dev_outputB);
	cudaDeviceSynchronize();
	cudaMemcpy(outputR, dev_outputR, RESOLUTION * (RESOLUTION / 4) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(outputG, dev_outputG, RESOLUTION * (RESOLUTION / 4) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(outputB, dev_outputB, RESOLUTION * (RESOLUTION / 4) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_tris);
	cudaFree(dev_outputR);
	cudaFree(dev_outputG);
	cudaFree(dev_outputB);
	Wrapper wrapper;
	wrapper.r = outputR;
	wrapper.g = outputG;
	wrapper.b = outputB;
	return wrapper;
}
float camPos[3] = { 0, 0, 0 };
float camRot[3] = { -0.0, 0, 0 };
float sunPos[3] = { 0.0f, 10.0f, 10.0f };
float mouseDiff[2] = { 0, 0 };
void mouseCallback(int event, int x, int y, int flags, void* userData)
{
	mouseDiff[0] = (float)x / RESOLUTION * glm::two_pi<float>() * 2;
	mouseDiff[1] = (float)y / RESOLUTION * glm::two_pi<float>() * 2;
}
void main()
{
	srand(time(NULL));
	Mesh mesh;
	mesh.triangles = std::vector<Triangle>(ALLOC_MEM_TRIS_NUM);
	loadFromObjectFile("C:/Users/arthu/ObjFiles/helmet.obj", mesh.triangles);
	cv::Mat canvas;

	glm::mat4 projection = glm::perspectiveFov(glm::half_pi<float>() / 2.0f, 2.0f, 2.0f, 0.01f, 100.0f);

	float frameCount = 0;
	glm::mat4 identity = glm::identity<glm::mat4>();

	float playerPos[3] = { 0, 2.0f, 0 };
	float playerVec[3] = { 0, 0, 0 };
	float playerRotY = 0.0f;
	curandState* state;
	float cameraDist = 4.0f;
	float depth_UNDERCOVER = 1.0f;
	int depth = 1.0f;
	oidn::DeviceRef device = oidn::newDevice();
	device.commit();
	oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
	while (true)
	{
		frameCount++;
		canvas = cv::Mat::zeros(cv::Size(RESOLUTION, RESOLUTION), CV_8UC3);
		std::vector<Triangle> oldTris = mesh.triangles;
		float s = clock();
		glm::quat rotQuat = glm::angleAxis(camRot[0], glm::vec3{ 1.0f, 0.0f, 0.0f });
		glm::mat4 rotMatX = glm::mat4_cast(rotQuat);
		rotQuat = glm::angleAxis(camRot[1], glm::vec3{ 0.0f, 1.0f, 0.0f });
		glm::mat4 rotMatY = glm::mat4_cast(rotQuat);
		rotQuat = glm::angleAxis(camRot[2], glm::vec3{ 0.0f, 0.0f, 1.0f });
		glm::mat4 rotMatZ = glm::mat4_cast(rotQuat);
		glm::vec4 lookVector = { 0, 0, 1, 0 };
		lookVector = rotMatX * rotMatY * rotMatZ * lookVector;

		// Camera physics

		std::vector<Triangle> tris = mesh.triangles;
		float closeT = cameraDist;
		float3 rayVec = normalize({ -cosf(playerRotY + glm::half_pi<float>()), 0.2f, -sinf(playerRotY + glm::half_pi<float>()) });

		for (int i = 0;i < ALLOC_MEM_TRIS_NUM;i++)
		{
			float3 p1{ tris[i].points[0], tris[i].points[1], tris[i].points[2] };
			float3 p2{ tris[i].points[3], tris[i].points[4], tris[i].points[5] };
			float3 p3{ tris[i].points[6], tris[i].points[7], tris[i].points[8] };
			float t;
			if (rayTriangleIntersect(p1, p2, p3, t, { playerPos[0], playerPos[1], playerPos[2] }, rayVec))
			{
				if (t < closeT)
				{
					closeT = t;
				}
			}
		}
		camPos[0] = playerPos[0] + max(closeT - 0.1f, 0.0f) * rayVec.x;
		camPos[1] = playerPos[1] + max(closeT - 0.1f, 0.0f) * rayVec.y;
		camPos[2] = playerPos[2] + max(closeT - 0.1f, 0.0f) * rayVec.z;

		// Player physics
		rayVec = { 0, -1, 0 };
		closeT = 0.2f;
		for (int i = 0;i < ALLOC_MEM_TRIS_NUM;i++)
		{
			float3 p1{ tris[i].points[0], tris[i].points[1], tris[i].points[2] };
			float3 p2{ tris[i].points[3], tris[i].points[4], tris[i].points[5] };
			float3 p3{ tris[i].points[6], tris[i].points[7], tris[i].points[8] };
			float t;
			if (rayTriangleIntersect(p1, p2, p3, t, { playerPos[0], playerPos[1], playerPos[2] }, rayVec))
			{
				if (t < closeT)
				{
					closeT = t;
					playerVec[0] = -tris[i].normal.x * 0.01f;
					playerVec[1] = -tris[i].normal.y * 0.015f;
					playerVec[2] = -tris[i].normal.z * 0.01f;
				}
			}
		}
		playerVec[1] -= 0.01f;
		playerVec[0] *= 0.97f;
		playerVec[2] *= 0.97f;
		playerPos[0] += playerVec[0];
		playerPos[1] += playerVec[1];
		playerPos[2] += playerVec[2];
		depth_UNDERCOVER += 0.5f;
		depth = (int)depth_UNDERCOVER;
		for (int y = 0;y < 4;y++)
		{
			Wrapper col = helper(y, camPos[0], camPos[1], camPos[2], sunPos[0], sunPos[1], sunPos[2], playerPos[0], playerPos[1], playerPos[2], camRot[0], camRot[1], &mesh.triangles[0], state, depth);
			for (int y2 = 0;y2 < RESOLUTION / 4;y2++)
			{
				for (int x = 0;x < RESOLUTION;x++)
				{

					cv::Vec3b& at = canvas.at<cv::Vec3b>(y2 + y * (RESOLUTION / 4), x);
					at.val[0] = col.r[(x + y2 * (RESOLUTION))];
					at.val[1] = col.g[(x + y2 * (RESOLUTION))];
					at.val[2] = col.b[(x + y2 * (RESOLUTION))];

				}
			}
		}
		mesh.triangles = oldTris;
		std::cout << (clock() - s) / CLOCKS_PER_SEC << std::endl;

		
		// Create a denoising filter
		if (depth_UNDERCOVER > 4.0f)
		{
			canvas.convertTo(canvas, CV_32FC3);
			filter.setImage("color", canvas.data, oidn::Format::Float3, RESOLUTION, RESOLUTION);
			filter.setImage("output", canvas.data, oidn::Format::Float3, RESOLUTION, RESOLUTION);
			filter.commit();
			filter.execute();
		}
		cv::imshow("Output", canvas);
		cv::setMouseCallback("Output", mouseCallback);
		cv::waitKey(1);
		if (GetKeyState('W') & 0x8000)
		{
			playerVec[2] += lookVector.z * 0.1;
			playerVec[0] += -lookVector.x * 0.1;
			playerVec[1] += lookVector.y * 0.1;
			depth_UNDERCOVER = 1.0f;
		}
		if (GetKeyState('S') & 0x8000)
		{
			playerVec[2] -= lookVector.z * 0.1;
			playerVec[0] -= -lookVector.x * 0.1;
			playerVec[1] -= lookVector.y * 0.1;
			depth_UNDERCOVER = 1.0f;
		}
		playerRotY = mouseDiff[0];
		camRot[1] = mouseDiff[0];
		if (GetKeyState('R') & 0x8000)
		{
			playerPos[1] -= 0.05f;
			depth_UNDERCOVER = 1.0f;

		}
		if (GetKeyState('T') & 0x8000)
		{
			playerPos[1] += 0.05f;
			depth_UNDERCOVER = 1.0f;
		}
		if (GetKeyState('F') & 0x8000)
		{
			sunPos[0] += 0.1f;
			depth_UNDERCOVER = 1.0f;
		}
		if (GetKeyState('G') & 0x8000)
		{
			sunPos[0] -= 0.1f;
			depth_UNDERCOVER = 1.0f;
		}
		if (GetKeyState('H') & 0x8000)
		{
			sunPos[1] += 0.1f;
			depth_UNDERCOVER = 1.0f;
		}
		if (GetKeyState('J') & 0x8000)
		{
			sunPos[1] -= 0.1f;
			depth_UNDERCOVER = 1.0f;
		}
		if (GetKeyState('K') & 0x8000)
		{
			sunPos[2] += 0.1f;
			depth_UNDERCOVER = 1.0f;
		}
		if (GetKeyState('L') & 0x8000)
		{
			sunPos[2] -= 0.1f;
			depth_UNDERCOVER = 1.0f;
		}
		if (GetKeyState('E') & 0x8000)
		{
			cameraDist -= 0.05f;
			depth_UNDERCOVER = 1.0f;

		}
		if (GetKeyState('Q') & 0x8000)
		{
			cameraDist += 0.05f;
			depth_UNDERCOVER = 1.0f;
		}
	}
	return;
}