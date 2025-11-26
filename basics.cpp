#include "precomp.h"
#include "basics.h"

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 1: basics. Link:
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics
// This is bare-bones BVH construction and traversal code, running in
// a minimalistic framework.
// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new BasicBVHApp(); }

// triangle count
#define N	64

// forward declarations
void Subdivide(uint nodeIdx);
void UpdateNodeBounds(uint nodeIdx);

// minimal structs
struct Tri { float3 vertex0, vertex1, vertex2; float3 centroid; };
__declspec(align(32)) struct BVHNode
{
	float3 aabbMin, aabbMax;
	uint leftFirst, triCount;
	bool isLeaf() { return triCount > 0; }
};
struct Ray { float3 O, D; float t = 1e30f; };

// application data
Tri tri[N];
uint triIdx[N];
BVHNode bvhNode[N * 2];
uint rootNodeIdx = 0, nodesUsed = 2;

// functions

// 1. Ray-Triangle Intersection: Möller-Trumbore algorithm
void IntersectTri(Ray& ray, const Tri& tri)
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;

	// Begin calculating determinant (a)
	const float3 h = cross(ray.D, edge2);
	const float a = dot(edge1, h);

	// Check for parallel ray (determinant near zero)
	if (a > -0.00001f && a < 0.00001f) return;

	const float f = 1.0f / a;
	const float3 s = ray.O - tri.vertex0;
	const float u = f * dot(s, h);

	// Check u barycentric coordinate
	if (u < 0 || u > 1) return;

	const float3 q = cross(s, edge1);
	const float v = f * dot(ray.D, q);

	// Check v barycentric coordinate (must be between 0 and 1-u)
	if (v < 0 || u + v > 1) return;

	// Calculate t (distance along the ray)
	const float t = f * dot(edge2, q);

	// Check if the intersection is in front of the ray and closer than the current hit
	if (t > 0.0001f) // 0.0001f is epsilon to avoid self-intersection
	{
		if (t < ray.t) ray.t = t;
	}
}

// 2. Ray-AABB Intersection: Slab Method (returns hit distance or a large float)
float IntersectAABB(const Ray& ray, const float3 bmin, const float3 bmax)
{
	float tmin = (bmin.x - ray.O.x) / ray.D.x;
	float tmax = (bmax.x - ray.O.x) / ray.D.x;
	if (tmin > tmax) { float t = tmin; tmin = tmax; tmax = t; } // Manual swap

	float tymin = (bmin.y - ray.O.y) / ray.D.y;
	float tymax = (bmax.y - ray.O.y) / ray.D.y;
	if (tymin > tymax) { float t = tymin; tmin = tymax; tmax = t; } // Manual swap

	if (tmin > tymax || tymin > tmax) return 1e30f;
	tmin = max(tmin, tymin);
	tmax = min(tmax, tymax);

	float tzmin = (bmin.z - ray.O.z) / ray.D.z;
	float tzmax = (bmax.z - ray.O.z) / ray.D.z;
	if (tzmin > tzmax) { float t = tzmin; tmin = tzmax; tmax = t; } // Manual swap

	if (tmin > tzmax || tzmin > tmax) return 1e30f;
	tmin = max(tmin, tzmin);
	tmax = min(tmax, tzmax);

	// Check if the intersection is valid (in front of ray and potentially closest hit)
	if (tmax < 0 || tmin > ray.t) return 1e30f;
	return tmin; // Return hit distance
}

// 3. BVH Traversal: Simple Recursive Method
void Intersect(Ray& ray, const uint nodeIdx)
{
	BVHNode& node = bvhNode[nodeIdx];

	// LEAF node
	if (node.isLeaf())
	{
		// Intersect all triangles in this leaf
		for (uint i = 0; i < node.triCount; i++)
		{
			uint triIdxInMesh = triIdx[node.leftFirst + i];
			IntersectTri(ray, tri[triIdxInMesh]);
		}
		return;
	}

	// INTERIOR node: check children's AABBs
	uint leftIdx = node.leftFirst;
	uint rightIdx = node.leftFirst + 1;

	BVHNode& leftChild = bvhNode[leftIdx];
	BVHNode& rightChild = bvhNode[rightIdx];

	// Intersect the AABBs of both children
	float dist1 = IntersectAABB(ray, leftChild.aabbMin, leftChild.aabbMax);
	float dist2 = IntersectAABB(ray, rightChild.aabbMin, rightChild.aabbMax);

	// Ordered traversal: Check the closer node first
	if (dist1 < dist2)
	{
		// Traverses the nearer child (left) if its box is hit (dist1 < ray.t)
		if (dist1 < ray.t) Intersect(ray, leftIdx);
		// Traverses the far child (right) ONLY IF its box is closer than the current best hit
		if (dist2 < ray.t) Intersect(ray, rightIdx);
	}
	else
	{
		// Traverses the nearer child (right)
		if (dist2 < ray.t) Intersect(ray, rightIdx);
		// Traverses the far child (left) ONLY IF its box is closer than the current best hit
		if (dist1 < ray.t) Intersect(ray, leftIdx);
	}
}

void UpdateNodeBounds(uint nodeIdx)
{
	BVHNode& node = bvhNode[nodeIdx];
	node.aabbMin = float3(1e30f);
	node.aabbMax = float3(-1e30f);
	for (uint i = 0; i < node.triCount; i++)
	{
		const Tri& leafTri = tri[triIdx[node.leftFirst + i]];
		node.aabbMin = fminf(node.aabbMin, leafTri.vertex0);
		node.aabbMin = fminf(node.aabbMin, leafTri.vertex1);
		node.aabbMin = fminf(node.aabbMin, leafTri.vertex2);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex0);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex1);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex2);
	}
}

void Subdivide(uint nodeIdx)
{
	// terminate recursion when max 2 triangles are left in a node
	BVHNode& node = bvhNode[nodeIdx];
	if (node.triCount <= 2) return;

	// determine split axis and position
	float3 extent = node.aabbMax - node.aabbMin;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;
	float splitPos = node.aabbMin[axis] + extent[axis] / 2;

	// in-place partition of the triangle index array
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		if (tri[triIdx[i]].centroid[axis] < splitPos)
			i++;
		else
			// manual swap
		{
			uint t = triIdx[i];
			triIdx[i] = triIdx[j];
			triIdx[j--] = t;
		}
	}

	// abort if one side is empty
	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return;

	// create child nodes
	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;
	bvhNode[leftChildIdx].leftFirst = node.leftFirst;
	bvhNode[leftChildIdx].triCount = leftCount;
	bvhNode[rightChildIdx].leftFirst = i;
	bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;
	UpdateNodeBounds(leftChildIdx);
	UpdateNodeBounds(rightChildIdx);

	// recurse
	Subdivide(leftChildIdx);
	Subdivide(rightChildIdx);
}

void CalculateTriangleCentroids()
{
	for (int i = 0; i < N; i++)
		tri[i].centroid = (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;
}

void BuildBVH()
{
	// setup initial indices
	for (int i = 0; i < N; i++) triIdx[i] = i;

	// calculate all centroids for the SAH-split logic
	CalculateTriangleCentroids();

	// set root node properties
	BVHNode& root = bvhNode[rootNodeIdx];
	root.leftFirst = 0;
	root.triCount = N;
	UpdateNodeBounds(rootNodeIdx);

	// start recursive subdivision
	Subdivide(rootNodeIdx);
}

// BasicBVHApp implementation

void BasicBVHApp::Init()
{
	// intialize a scene with N random triangles
	for (int i = 0; i < N; i++)
	{
		float3 r0 = float3(RandomFloat(), RandomFloat(), RandomFloat());
		float3 r1 = float3(RandomFloat(), RandomFloat(), RandomFloat());
		float3 r2 = float3(RandomFloat(), RandomFloat(), RandomFloat());
		tri[i].vertex0 = r0 * 9 - float3(5);
		tri[i].vertex1 = tri[i].vertex0 + r1, tri[i].vertex2 = tri[i].vertex0 + r2;
	}
	// construct the BVH
	BuildBVH();
}

void BasicBVHApp::Tick(float deltaTime)
{
	// draw the scene
	screen->Clear(0);
	// define the corners of the screen in worldspace
	float3 p0(-1, 1, -15), p1(1, 1, -15), p2(-1, -1, -15);
	Ray ray;
	Timer t;
	for (int y = 0; y < SCRHEIGHT; y++) for (int x = 0; x < SCRWIDTH; x++)
	{
		// calculate the position of a pixel on the screen in worldspace
		float3 pixelPos = p0 + (p1 - p0) * (x / (float)SCRWIDTH) +
			(p2 - p0) * (y / (float)SCRHEIGHT);
		ray.O = float3(0, 0, 0);
		ray.D = normalize(pixelPos - ray.O);

		// --- The traversal call ---
		ray.t = 1e30f; // Reset ray distance for new ray
		Intersect(ray, rootNodeIdx); // Start traversal from the root node

		// If ray.t < 1e30f, we hit something; draw a color
		if (ray.t < 1e30f) screen->Plot(x, y, 0xff0000);
		else screen->Plot(x, y, 0);
	}
	float t_ms = t.elapsed() * 1000;
	// uncomment to see BVH build/traverse time
	// printf( "Build: %.2fms, Traverse: %.2fms\n", build_t_ms, t_ms );
}