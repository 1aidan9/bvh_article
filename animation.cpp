#include "precomp.h"
#include "animation.h"

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 4: animation.
// This version shows how to ray trace an animated mesh using
// two methods: rebuilding and refitting. The refitting method is
// good for changes that do not change the topology of the mesh.
// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new AnimationApp(); }

// enable the use of SSE in the AABB intersection function
#define USE_SSE

// triangle count
#define N	20944 // hardcoded for the big-ben mesh

// bin count
#define BINS 8

// forward declarations
void Subdivide(uint nodeIdx);
void UpdateNodeBounds(uint nodeIdx);

// application data (global for simplicity in a single-file BVH implementation)
Tmpl8::Tri tri[N], original[N];
uint triIdx[N];
Tmpl8::BVHNode* bvhNode = (Tmpl8::BVHNode*)_aligned_malloc(sizeof(Tmpl8::BVHNode) * N * 2, 64);
uint rootNodeIdx = 0, nodesUsed = 2;

// functions

void IntersectTri(Tmpl8::Ray& ray, const Tmpl8::Tri& tri)
{
	// Moeller-Trumbore ray/triangle intersection algorithm
	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;
	const float3 h = cross(ray.D, edge2);
	const float a = dot(edge1, h);
	if (fabs(a) < 0.00001f) return; // ray parallel to triangle
	const float f = 1 / a;
	const float3 s = ray.O - tri.vertex0;
	const float u = f * dot(s, h);
	if (u < 0 || u > 1) return;
	const float3 q = cross(s, edge1);
	const float v = f * dot(ray.D, q);
	if (v < 0 || u + v > 1) return;
	const float t = f * dot(edge2, q);
	if (t > 0.0001f && t < ray.t) ray.t = t;
}

float IntersectAABB(const Tmpl8::Ray& ray, const Tmpl8::BVHNode* node)
{
#ifdef USE_SSE
	// uses precalculated 1/D in ray
	__m128 t1 = _mm_mul_ps(_mm_sub_ps(node->aabbMin4, ray.O4), ray.rD4);
	__m128 t2 = _mm_mul_ps(_mm_sub_ps(node->aabbMax4, ray.O4), ray.rD4);
	__m128 vmax = _mm_max_ps(t1, t2), vmin = _mm_min_ps(t1, t2);
	float tmax = min(vmax.m128_f32[0], min(vmax.m128_f32[1], vmax.m128_f32[2]));
	float tmin = max(vmin.m128_f32[0], max(vmin.m128_f32[1], vmin.m128_f32[2]));
	if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
#else
	float tx1 = (node->aabbMin.x - ray.O.x) * ray.rD.x;
	float tx2 = (node->aabbMax.x - ray.O.x) * ray.rD.x;
	float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
	float ty1 = (node->aabbMin.y - ray.O.y) * ray.rD.y;
	float ty2 = (node->aabbMax.y - ray.O.y) * ray.rD.y;
	tmin = max(tmin, min(ty1, ty2));
	tmax = min(tmax, max(ty1, ty2));
	float tz1 = (node->aabbMin.z - ray.O.z) * ray.rD.z;
	float tz2 = (node->aabbMax.z - ray.O.z) * ray.rD.z;
	tmin = max(tmin, min(tz1, tz2));
	tmax = min(tmax, max(tz1, tz2));
	if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
#endif
}

void IntersectBVH(Tmpl8::Ray& ray)
{
	// stackless traversal
	ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
	Tmpl8::BVHNode* node = &bvhNode[rootNodeIdx], * stack[64];
	uint stackPtr = 0;
	while (1)
	{
		if (node->isLeaf())
		{
			for (uint i = 0; i < node->triCount; i++)
				IntersectTri(ray, tri[triIdx[node->leftFirst + i]]);
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		Tmpl8::BVHNode* child1 = &bvhNode[node->leftFirst];
		Tmpl8::BVHNode* child2 = &bvhNode[node->leftFirst + 1];
		float dist1 = IntersectAABB(ray, child1);
		float dist2 = IntersectAABB(ray, child2);
		if (dist1 > dist2) { swap(dist1, dist2); swap(child1, child2); }
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = child1;
			if (dist2 != 1e30f) stack[stackPtr++] = child2;
		}
	}
}

// BVH Refitting (new for this project)
// Updates the AABBs from the leaves upwards without changing the BVH structure.
void AnimationApp::RefitBVH()
{
	Timer t;
	// iterate over all nodes and recompute bounds
	for (uint i = nodesUsed - 1; i >= 0; i--)
	{
		Tmpl8::BVHNode* node = &bvhNode[i];
		if (node->isLeaf())
		{
			// leaf: calculate bounds from primitives
			Tmpl8::aabb leafBounds;
			for (uint i = 0; i < node->triCount; i++)
			{
				const Tmpl8::Tri& triangle = tri[triIdx[node->leftFirst + i]];
				leafBounds.grow(triangle.vertex0);
				leafBounds.grow(triangle.vertex1);
				leafBounds.grow(triangle.vertex2);
			}
			node->aabbMin = leafBounds.bmin;
			node->aabbMax = leafBounds.bmax;
		}
		else
		{
			// interior node: merge child bounds
			Tmpl8::BVHNode* leftChild = &bvhNode[node->leftFirst];
			Tmpl8::BVHNode* rightChild = &bvhNode[node->leftFirst + 1];
			node->aabbMin = fminf(leftChild->aabbMin, rightChild->aabbMin);
			node->aabbMax = fmaxf(leftChild->aabbMax, rightChild->aabbMax);
		}
	}
	printf("Refit time: %.2fms\n", t.elapsed() * 1000);
}

// BVH construction (re-used from previous projects)

float CalculateNodeCost(const Tmpl8::BVHNode* node)
{
	float3 e = node->aabbMax - node->aabbMin;
	return e.x * e.y + e.y * e.z + e.z * e.x; // Surface Area
}

float FindBestSplitPlane(Tmpl8::BVHNode& node, int& axis, float& splitPos)
{
	float bestCost = 1e30f;
	float nodeArea = CalculateNodeCost(&node);

	// calculate node bounds and centroid bounds
	Tmpl8::aabb bounds;
	Tmpl8::aabb centroidBounds;
	for (uint i = 0; i < node.triCount; i++)
	{
		Tmpl8::Tri& tri = ::tri[triIdx[node.leftFirst + i]];
		bounds.grow(tri.vertex0);
		bounds.grow(tri.vertex1);
		bounds.grow(tri.vertex2);
		centroidBounds.grow(tri.centroid);
	}
	if (centroidBounds.bmin.x == 1e30f) return 1e30f;

	int bestAxis = -1;
	for (int a = 0; a < 3; a++) // try all three axes
	{
		if (centroidBounds.bmax[a] == centroidBounds.bmin[a]) continue;

		// populate bins
		static Tmpl8::Bin bins[BINS];
		for (int i = 0; i < BINS; i++) bins[i] = Tmpl8::Bin();
		float scale = BINS / (centroidBounds.bmax[a] - centroidBounds.bmin[a]);
		for (uint i = 0; i < node.triCount; i++)
		{
			Tmpl8::Tri& triangle = ::tri[triIdx[node.leftFirst + i]];
			int binIdx = min(BINS - 1, (int)((triangle.centroid[a] - centroidBounds.bmin[a]) * scale));
			bins[binIdx].triCount++;
			bins[binIdx].bounds.grow(triangle.vertex0);
			bins[binIdx].bounds.grow(triangle.vertex1);
			bins[binIdx].bounds.grow(triangle.vertex2);
		}

		// gather data for splits
		float cost[BINS - 1];
		for (int i = 0; i < BINS - 1; i++)
		{
			Tmpl8::aabb leftBounds, rightBounds;
			int leftCount = 0, rightCount = 0;
			for (int j = 0; j <= i; j++) leftBounds.grow(bins[j].bounds), leftCount += bins[j].triCount;
			for (int j = i + 1; j < BINS; j++) rightBounds.grow(bins[j].bounds), rightCount += bins[j].triCount;
			float costL = leftCount * leftBounds.area();
			float costR = rightCount * rightBounds.area();
			cost[i] = costL + costR;
		}

		// find best split plane
		for (int i = 0; i < BINS - 1; i++)
		{
			if (cost[i] < bestCost)
			{
				bestCost = cost[i];
				bestAxis = a;
				splitPos = centroidBounds.bmin[a] + (i + 1) * (centroidBounds.bmax[a] - centroidBounds.bmin[a]) / BINS;
			}
		}
	}

	axis = bestAxis;
	return bestCost / nodeArea;
}

void Subdivide(uint nodeIdx)
{
	Tmpl8::BVHNode& node = bvhNode[nodeIdx];

	// SAH: find best split plane
	int axis;
	float splitPos;
	float cost = FindBestSplitPlane(node, axis, splitPos);
	float splitCost = node.triCount;
	if (cost >= splitCost) return; // splitting is not worth it, stop here

	// split primitives
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		if (::tri[triIdx[i]].centroid[axis] < splitPos)
			i++;
		else
			swap(triIdx[i], triIdx[j--]);
	}

	// create child nodes
	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return;

	// create left child
	uint leftChildIdx = nodesUsed++;
	bvhNode[leftChildIdx].leftFirst = node.leftFirst;
	bvhNode[leftChildIdx].triCount = leftCount;
	UpdateNodeBounds(leftChildIdx);
	Subdivide(leftChildIdx);

	// create right child
	uint rightChildIdx = nodesUsed++;
	bvhNode[rightChildIdx].leftFirst = i;
	bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
	UpdateNodeBounds(rightChildIdx);
	Subdivide(rightChildIdx);

	// current node becomes interior
	node.leftFirst = leftChildIdx;
	node.triCount = 0; // mark as interior
}

void UpdateNodeBounds(uint nodeIdx)
{
	Tmpl8::BVHNode& node = bvhNode[nodeIdx];
	node.aabbMin = float3(1e30f);
	node.aabbMax = float3(-1e30f);
	for (uint i = 0; i < node.triCount; i++)
	{
		uint triIdxInList = triIdx[node.leftFirst + i];
		const Tmpl8::Tri& triangle = ::tri[triIdxInList];
		node.aabbMin = fminf(node.aabbMin, triangle.vertex0);
		node.aabbMin = fminf(node.aabbMin, triangle.vertex1);
		node.aabbMin = fminf(node.aabbMin, triangle.vertex2);
		node.aabbMax = fmaxf(node.aabbMax, triangle.vertex0);
		node.aabbMax = fmaxf(node.aabbMax, triangle.vertex1);
		node.aabbMax = fmaxf(node.aabbMax, triangle.vertex2);
	}
}

void AnimationApp::BuildBVH()
{
	Timer t;
	// reset node counter
	nodesUsed = 2;

	// populate root node
	Tmpl8::BVHNode& root = bvhNode[rootNodeIdx];
	root.leftFirst = 0;
	root.triCount = N;

	// initialize triangle indices
	for (int i = 0; i < N; i++) triIdx[i] = i;

	// calculate bounds of the root node
	UpdateNodeBounds(rootNodeIdx);

	// start subdivision process
	Subdivide(rootNodeIdx);

	printf("BVH build time: %.2fms\n", t.elapsed() * 1000);
}

void AnimationApp::Animate()
{
	static float frame = 0;
	float angle = (frame++ * 0.01f);
	float s = sinf(angle), c = cosf(angle);
	for (int i = 0; i < N; i++)
	{
		// rotate vertices around Z axis
		Tmpl8::Tri& o = original[i];
		for (int j = 0; j < 3; j++)
		{
			float3& v = (&o.vertex0)[j];
			float x = v.x * c - v.y * s;
			float y = v.x * s + v.y * c;
			tri[i].centroid.x += x; tri[i].centroid.y += y; tri[i].centroid.z += v.z;
			(&tri[i].vertex0)[j] = float3(x, y, v.z);
		}
		// calculate new centroid
		tri[i].centroid = (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * (1.0f / 3.0f);
	}
}

// AnimationApp implementation

void AnimationApp::Init()
{
	// load Big Ben mesh
	FILE* file = fopen("assets/bigben.tri", "r");
	if (!file) { printf("Error: Could not open assets/bigben.tri\n"); return; }
	for (int t = 0; t < N; t++) fscanf(file, "%f %f %f %f %f %f %f %f %f\n",
		&original[t].vertex0.x, &original[t].vertex0.y, &original[t].vertex0.z,
		&original[t].vertex1.x, &original[t].vertex1.y, &original[t].vertex1.z,
		&original[t].vertex2.x, &original[t].vertex2.y, &original[t].vertex2.z);
	fclose(file);

	// copy original to current and calculate initial centroids
	for (int t = 0; t < N; t++)
	{
		tri[t] = original[t];
		tri[t].centroid = (tri[t].vertex0 + tri[t].vertex1 + tri[t].vertex2) * (1.0f / 3.0f);
	}

	// initialize animation and BVH
	Animate();
	BuildBVH();
}

void AnimationApp::Tick(float deltaTime)
{
	Animate();
	// BuildBVH(); // Uncomment this line to use the full rebuild method (slower)
	RefitBVH();   // Use RefitBVH for faster update (current default)

	// draw the scene: multithreaded tiles
	float3 p0(-1, 1, 2), p1(1, 1, 2), p2(-1, -1, 2); // Screen plane corners
	float3 camPos(0, 0, -3);

#pragma omp parallel for schedule(dynamic)
	for (int tile = 0; tile < (SCRWIDTH * SCRHEIGHT / 64); tile++)
	{
		// render an 8x8 tile
		int x = tile % (SCRWIDTH / 8), y = tile / (SCRWIDTH / 8);

		for (int v = 0; v < 8; v++) for (int u = 0; u < 8; u++)
		{
			// setup a primary ray
			Tmpl8::Ray ray;
			ray.O = camPos;
			float3 pixelPos = camPos + p0 +
				(p1 - p0) * ((x * 8 + u + 0.5f) / SCRWIDTH) +
				(p2 - p0) * ((y * 8 + v + 0.5f) / SCRHEIGHT);
			ray.D = normalize(pixelPos - ray.O);

			// trace ray
			IntersectBVH(ray);

			// write to screen
			uint pixel = (x * 8 + u) + (y * 8 + v) * SCRWIDTH;
			if (ray.t == 1e30f)
				screen->pixels[pixel] = 0; // Black for miss
			else
			{
				// Simple shading based on hit distance
				float t = min(ray.t * 0.1f, 1.0f);
				int c = (int)(t * 255);
				screen->pixels[pixel] = (c << 16) | (c << 8) | c; // Grayscale proportional to distance
			}
		}
	}
}