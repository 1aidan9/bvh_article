#include "precomp.h"
#include "faster.h"

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 2: faster rays.
// This version improves ray traversal speed using ordered traversal
// and SSE-aligned BVH nodes for fast AABB intersection.

TheApp* CreateApp() { return new FasterRaysApp(); }

// enable the use of SSE in the AABB intersection function
#define USE_SSE

// triangle count
#define N	12582 // hardcoded for the unity vehicle mesh

// --- Minimal Structs and Data ---

// The SSE-optimized Ray struct is critical for fast AABB intersection in BVH traversal.
__declspec(align(64)) struct Ray
{
	Ray() { O4 = D4 = rD4 = _mm_set1_ps(1); }
	union { struct { float3 O; float dummy1; }; __m128 O4; };
	union { struct { float3 D; float dummy2; }; __m128 D4; };
	union { struct { float3 rD; float dummy3; }; __m128 rD4; }; // Reciprocal direction for fast AABB check
	float t = 1e30f; // Ray hit distance
};

// 32-byte BVH node struct, SSE-aligned for fast AABB intersection.
struct BVHNode
{
	union { struct { float3 aabbMin; uint leftFirst; }; __m128 aabbMin4; };
	union { struct { float3 aabbMax; uint triCount; }; __m128 aabbMax4; };
	bool isLeaf() { return triCount > 0; }
	float area()
	{
		float3 e = aabbMax - aabbMin;
		return e.x * e.y + e.y * e.z + e.z * e.x;
	}
};

struct Tri { float3 vertex0, vertex1, vertex2; float3 centroid; };

Tri tri[N];
uint triIdx[N];
BVHNode bvhNode[N * 2];
uint rootNodeIdx = 0, nodesUsed = 2;

// forward declarations
void Subdivide(uint nodeIdx);
void UpdateNodeBounds(uint nodeIdx);

// --- BVH Intersection Functions ---

// Fast AABB intersection using reciprocal direction (ray.rD)
inline float IntersectAABB(const Ray& ray, const float3 bmin, const float3 bmax)
{
	float tx1 = (bmin.x - ray.O.x) * ray.rD.x, tx2 = (bmax.x - ray.O.x) * ray.rD.x;
	float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
	float ty1 = (bmin.y - ray.O.y) * ray.rD.y, ty2 = (bmax.y - ray.O.y) * ray.rD.y;
	tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
	float tz1 = (bmin.z - ray.O.z) * ray.rD.z, tz2 = (bmax.z - ray.O.z) * ray.rD.z;
	tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
	if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
}

// SSE Optimized AABB intersection
inline float IntersectAABB_SSE(const Ray& ray, const __m128& bmin4, const __m128& bmax4)
{
	static __m128 mask4 = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_set_ps(1, 0, 0, 0));
	__m128 t1 = _mm_mul_ps(_mm_sub_ps(_mm_and_ps(bmin4, mask4), ray.O4), ray.rD4);
	__m128 t2 = _mm_mul_ps(_mm_sub_ps(_mm_and_ps(bmax4, mask4), ray.O4), ray.rD4);
	__m128 vmax4 = _mm_max_ps(t1, t2), vmin4 = _mm_min_ps(t1, t2);
	float tmax = min(vmax4.m128_f32[0], min(vmax4.m128_f32[1], vmax4.m128_f32[2]));
	float tmin = max(vmin4.m128_f32[0], max(vmin4.m128_f32[1], vmin4.m128_f32[2]));
	if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
}

// Moeller-Trumbore ray/triangle intersection
void IntersectTri(Ray& ray, const Tri& tri)
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;
	const float3 h = cross(ray.D, edge2);
	const float a = dot(edge1, h);
	if (a > -0.00001f && a < 0.00001f) return; // ray parallel to triangle
	const float f = 1 / a;
	const float3 s = ray.O - tri.vertex0;
	const float u = f * dot(s, h);
	if (u < 0 || u > 1) return;
	const float3 q = cross(s, edge1);
	const float v = f * dot(ray.D, q);
	if (v < 0 || u + v > 1) return;
	const float t = f * dot(edge2, q);
	if (t > 0.00001f && t < ray.t) ray.t = t;
}

// Ordered BVH Traversal (using a local stack)
void IntersectBVH(Ray& ray)
{
	// Calculate reciprocal ray directions (needed for fast AABB check)
	ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);

	BVHNode* stack[64];
	uint stackPtr = 0;
	BVHNode* node = &bvhNode[rootNodeIdx];

	while (1)
	{
		if (node->isLeaf())
		{
			// Leaf node: intersect triangles
			for (uint i = 0; i < node->triCount; i++)
				IntersectTri(ray, tri[triIdx[node->leftFirst + i]]);

			// Pop from stack or terminate
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			// Interior node: check children
			BVHNode* child1 = &bvhNode[node->leftFirst];
			BVHNode* child2 = &bvhNode[node->leftFirst + 1];

#ifdef USE_SSE
			float dist1 = IntersectAABB_SSE(ray, child1->aabbMin4, child1->aabbMax4);
			float dist2 = IntersectAABB_SSE(ray, child2->aabbMin4, child2->aabbMax4);
#else
			float dist1 = IntersectAABB(ray, child1->aabbMin, child1->aabbMax);
			float dist2 = IntersectAABB(ray, child2->aabbMin, child2->aabbMax);
#endif

			// Ordered Traversal: swap children so child1 is the *closer* one
			if (dist1 > dist2)
			{
				swap(dist1, dist2);
				swap(child1, child2);
			}

			if (dist1 == 1e30f)
			{
				// No hit on either child: pop from stack
				if (stackPtr == 0) break; else node = stack[--stackPtr];
			}
			else
			{
				// Closest child (child1) hit: move to that node.
				node = child1;
				// If farther child (child2) also hit: push it to the stack
				if (dist2 != 1e30f) stack[stackPtr++] = child2;
			}
		}
	}
}

// --- BVH Build Functions (Simple SAH / Object Split) ---

void UpdateNodeBounds(uint nodeIdx)
{
	BVHNode& node = bvhNode[nodeIdx];
	node.aabbMin = float3(1e30f);
	node.aabbMax = float3(-1e30f);
	for (uint i = 0; i < node.triCount; i++)
	{
		Tri& leafTri = tri[triIdx[node.leftFirst + i]];
		// The BVH node must contain the entire triangle
		node.aabbMin = fminf(node.aabbMin, leafTri.vertex0);
		node.aabbMin = fminf(node.aabbMin, leafTri.vertex1);
		node.aabbMin = fminf(node.aabbMin, leafTri.vertex2);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex0);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex1);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.vertex2);
	}
}

// Evaluate Split Cost using SAH
float EvaluateSAH(BVHNode& node, int axis, float pos)
{
	// determine triangle counts and bounds for this split candidate
	float3 leftMin = float3(1e30f), leftMax = float3(-1e30f);
	float3 rightMin = float3(1e30f), rightMax = float3(-1e30f);
	int leftCount = 0, rightCount = 0;
	for (uint i = 0; i < node.triCount; i++)
	{
		Tri& triangle = tri[triIdx[node.leftFirst + i]];
		// FIX: Access vector component using pointer arithmetic to avoid operator[] errors
		if ((&triangle.centroid.x)[axis] < pos)
		{
			leftCount++;
			leftMin = fminf(leftMin, triangle.vertex0); leftMax = fmaxf(leftMax, triangle.vertex0);
			leftMin = fminf(leftMin, triangle.vertex1); leftMax = fmaxf(leftMax, triangle.vertex1);
			leftMin = fminf(leftMin, triangle.vertex2); leftMax = fmaxf(leftMax, triangle.vertex2);
		}
		else
		{
			rightCount++;
			rightMin = fminf(rightMin, triangle.vertex0); rightMax = fmaxf(rightMax, triangle.vertex0);
			rightMin = fminf(rightMin, triangle.vertex1); rightMax = fmaxf(rightMax, triangle.vertex1);
			rightMin = fminf(rightMin, triangle.vertex2); rightMax = fmaxf(rightMax, triangle.vertex2);
		}
	}
	float3 le = leftMax - leftMin;
	float3 re = rightMax - rightMin;
	float cost = leftCount * (le.x * le.y + le.y * le.z + le.z * le.x) +
		rightCount * (re.x * re.y + re.y * re.z + re.z * re.x);
	return cost > 0 ? cost : 1e30f;
}

void Subdivide(uint nodeIdx)
{
	BVHNode& node = bvhNode[nodeIdx];

	// determine split axis using SAH
	int bestAxis = -1;
	float bestPos = 0, bestCost = 1e30f;

	// We will test the centroid of each triangle as a potential split plane
	for (int axis = 0; axis < 3; axis++)
	{
		for (uint i = 0; i < node.triCount; i++)
		{
			Tri& triangle = tri[triIdx[node.leftFirst + i]];
			// FIX: Access vector component using pointer arithmetic
			float candidatePos = (&triangle.centroid.x)[axis];
			float cost = EvaluateSAH(node, axis, candidatePos);
			if (cost < bestCost)
			{
				bestPos = candidatePos;
				bestAxis = axis;
				bestCost = cost;
			}
		}
	}

	float3 e = node.aabbMax - node.aabbMin;
	float parentArea = e.x * e.y + e.y * e.z + e.z * e.x;
	float parentCost = node.triCount * parentArea;

	if (bestCost >= parentCost) return; // Terminate if split doesn't improve cost

	int axis = bestAxis;
	float splitPos = bestPos;

	// in-place partition
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		// FIX: Access vector component using pointer arithmetic
		if ((&tri[triIdx[i]].centroid.x)[axis] < splitPos)
			i++;
		else
			swap(triIdx[i], triIdx[j--]);
	}

	// abort split if one side is empty
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
	node.triCount = 0; // mark as interior node

	// calculate child bounds and recurse
	UpdateNodeBounds(leftChildIdx);
	UpdateNodeBounds(rightChildIdx);
	Subdivide(leftChildIdx);
	Subdivide(rightChildIdx);
}

void BuildBVH()
{
	// calculate triangle centroids for construction
	for (int i = 0; i < N; i++)
		tri[i].centroid = (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;

	// set initial node in the root
	bvhNode[rootNodeIdx].leftFirst = 0;
	bvhNode[rootNodeIdx].triCount = N;
	// initialize triangle indices
	for (int i = 0; i < N; i++) triIdx[i] = i;

	// subdivide root node
	UpdateNodeBounds(rootNodeIdx);
	Timer t;
	Subdivide(rootNodeIdx);
	printf("BVH construction in %.2fms\n", t.elapsed() * 1000);
}

// --- Application Implementation ---

void FasterRaysApp::Init()
{
	// load the model data
	FILE* file = fopen("assets/unity.tri", "r");
	if (!file)
	{
		printf("Error: 'assets/unity.tri' not found.\n");
		return;
	}
	float a, b, c, d, e, f, g, h, i;
	for (int t = 0; t < N; t++)
	{
		fscanf(file, "%f %f %f %f %f %f %f %f %f\n",
			&a, &b, &c, &d, &e, &f, &g, &h, &i);
		tri[t].vertex0 = float3(a, b, c);
		tri[t].vertex1 = float3(d, e, f);
		tri[t].vertex2 = float3(g, h, i);
	}
	fclose(file);

	// construct the BVH
	BuildBVH();
}

void FasterRaysApp::Tick(float deltaTime)
{
	// draw the scene
	screen->Clear(0);
	// define the corners of the screen in worldspace
	float3 p0(-2.5f, 0.8f, -0.5f), p1(-0.5f, 0.8f, -0.5f), p2(-2.5f, -1.2f, -0.5f);
	Ray ray;
	Timer t;

	// Camera position
	ray.O = float3(-1.5f, -0.2f, -1.5f);

	// render tiles of pixels
	for (int y = 0; y < SCRHEIGHT; y += 4) for (int x = 0; x < SCRWIDTH; x += 4)
	{
		// render a single tile
		for (int v = 0; v < 4; v++) for (int u = 0; u < 4; u++)
		{
			// setup a primary ray
			float3 pixelPos = p0 + (p1 - p0) * ((x + u) / (float)SCRWIDTH) + (p2 - p0) * ((y + v) / (float)SCRHEIGHT);
			ray.D = normalize(pixelPos - ray.O);
			ray.t = 1e30f;

			// trace the ray and shade the pixel
			IntersectBVH(ray);

			if (ray.t < 1e30f)
			{
				// hit: visualize intersection distance
				float3 I = ray.O + ray.D * ray.t;
				float d = (I.x + I.y + I.z) * 0.1f;
				uint c = (int)(d * 256);
				screen->Plot(x + u, y + v, c * 0x10000 + c * 0x100 + c);
			}
			else
				screen->Plot(x + u, y + v, 0);
		}
	}
	printf("Raytracing: %.2fms\n", t.elapsed() * 1000);
}