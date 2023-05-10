//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

//A feladatot a Minimál CPU sugárkövetõ példaprogramból kezdtem el megoldani.
//https://edu.vik.bme.hu/mod/url/view.php?id=97492

#include "framework.h"

const float epsilon = 0.0001f;

struct Hit {
	float t;
	vec3 position, normal;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) : start(_start), dir(normalize(_dir)) {}
};

class Intersectable {
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Triangle : public Intersectable {
	vec3 r1, r2, r3, n;

	Triangle(const vec3& _r1, const vec3& _r2, const vec3& _r3)
		:r1(_r1), r2(_r2), r3(_r3), n(normalize(cross(r2 - r1, r3 - r1))) {}

	Hit intersect(const Ray& ray) {
		Hit hit;
		hit.t = dot(r1 - ray.start, n) / dot(ray.dir, n);
		if (hit.t <= 0) return Hit();
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = n;
		if ((dot(cross(r2-r1, hit.position-r1), n) > 0) &&
		    (dot(cross(r3-r2, hit.position-r2), n) > 0) &&
		    (dot(cross(r1-r3, hit.position-r3), n) > 0)) return hit;
		return Hit();
	}
};

struct Cone : public Intersectable {
	vec3 p, n;
	float h, alpha;

	Cone(const vec3& _p, float _h, const vec3& _axis, float _alpha)
		:p(_p), h(_h), alpha(_alpha * M_PI / 180.0f),
		 n(normalize(_axis)) {}

	Hit intersect(const Ray& ray) {
		float a = dot(ray.dir, n) * dot(ray.dir, n) - dot(ray.dir, ray.dir) * cosf(alpha) * cosf(alpha);
		float b = 2 * (dot(ray.dir, n) * dot(ray.start - p, n) - dot(ray.dir, ray.start - p) * cosf(alpha) * cosf(alpha));
		float c = dot(ray.start - p, n) * dot(ray.start - p, n) - dot(ray.start - p, ray.start - p) * cosf(alpha) * cosf(alpha);
		float discr = b * b - 4. * a * c;
		if (discr < 0) return Hit();
		float t1 = (-b - sqrt(discr)) / (2 * a);
		float t2 = (-b + sqrt(discr)) / (2 * a);

		float t;
		if (t1 < 0 && t2 < 0) return Hit();
		else if (t1 < 0 && t2 > 0) t = t2;
		else if (t1 > 0 && t2 < 0) t = t1;
		else t1 < t2 ? t = t1 : t = t2;
		 
		vec3 p1 = ray.start + t1 * ray.dir;
		vec3 p2 = ray.start + t2 * ray.dir;
		vec3 pos;
		if (length(p-p2) > h && length(p-p1) < h) pos = p1;
		else if (length(p-p2) < h && length(p-p1) > h) pos = p2;
		else if (length(p-p2) < length(p-p1)) pos = p1;
		else pos = p2;

		if (dot(pos - p, n) < 0 || dot(pos - p, n) > h) return Hit();

		Hit hit;
		hit.t = t;
		hit.normal = normalize(pos * dot(n, pos) / dot(pos, pos) - n);
		return hit;
	}
};

//Source: https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html
struct Cube {
	std::vector<vec3> vtx;
	vec3 center = vec3(0.0f, 0.0f, 0.0f);
	float length = 0.2f, dist = sqrt(3) * length / 2;
public:
	Cube() { build(); }

	Cube(vec3 _center, float _length)
		:center(_center), length(_length) { build(); }

	vec3 getCenter() { return center; }

	void build() {
		vtx.clear();

		vtx.push_back(vec3(center.x - dist, center.y - dist, center.z - dist));
		vtx.push_back(vec3(center.x + dist, center.y - dist, center.z - dist));
		vtx.push_back(vec3(center.x - dist, center.y - dist, center.z + dist));
		vtx.push_back(vec3(center.x + dist, center.y - dist, center.z + dist));
		vtx.push_back(vec3(center.x - dist, center.y + dist, center.z - dist));
		vtx.push_back(vec3(center.x + dist, center.y + dist, center.z - dist));
		vtx.push_back(vec3(center.x - dist, center.y + dist, center.z + dist));
		vtx.push_back(vec3(center.x + dist, center.y + dist, center.z + dist));
	}

	std::vector<Intersectable*> create(std::vector<Intersectable*> objects) {
		objects.push_back(new Triangle(vtx[0], vtx[1], vtx[2]));
		objects.push_back(new Triangle(vtx[0], vtx[1], vtx[5]));
		objects.push_back(new Triangle(vtx[0], vtx[2], vtx[6]));
		objects.push_back(new Triangle(vtx[0], vtx[4], vtx[5]));
		objects.push_back(new Triangle(vtx[0], vtx[4], vtx[6]));
		objects.push_back(new Triangle(vtx[1], vtx[2], vtx[3]));
		objects.push_back(new Triangle(vtx[4], vtx[5], vtx[6]));
		objects.push_back(new Triangle(vtx[5], vtx[6], vtx[7]));
		//objects.push_back(new Triangle(vtx[1], vtx[3], vtx[7]));
		//objects.push_back(new Triangle(vtx[1], vtx[5], vtx[7]));
		//objects.push_back(new Triangle(vtx[2], vtx[3], vtx[6]));
		//objects.push_back(new Triangle(vtx[3], vtx[6], vtx[7]));

		return objects;
	}

	void print() {
		printf("Center: %f %f %f\n", center.x, center.y, center.z);
		printf("Vertices:\n");
		for (auto& v : vtx) {
			printf("%f %f %f\n", v.x, v.y, v.z);
		}
	}

	void moveTo(vec3 _center) {
		vec3 oldCenter = center;
		center = _center;
		vec3 newCenter = oldCenter - center;

		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x - newCenter.x, oldv.y - newCenter.y, oldv.z - newCenter.z);
		}
	}

	void rotateX(float angle) {
		angle = angle * (M_PI / 180.0f);
		vec3 currentCenter = center;
		moveTo(vec3(0.0f, 0.0f, 0.0f));
		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x + currentCenter.x,
					 oldv.y * cosf(angle) - oldv.z * sinf(angle) + currentCenter.y,
					 oldv.z * cosf(angle) + oldv.y * sinf(angle) + currentCenter.z);
		}
	}

	void rotateY(float angle) {
		angle = angle * (M_PI / 180.0f);
		vec3 currentCenter = center;
		moveTo(vec3(0.0f, 0.0f, 0.0f));
		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x * cosf(angle) - oldv.z * sinf(angle) + currentCenter.x,
					 oldv.y + currentCenter.y,
					 oldv.z * cosf(angle) + oldv.x * sinf(angle) + currentCenter.z);
		}
	}
};

//Source: https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html
struct Octahedron {
	std::vector<vec3> vtx;
	vec3 center = vec3(0.0f, 0.0f, 0.0f);
	float scale = 0.2f;

	Octahedron() { build(); }

	Octahedron(vec3 _center, float _scale)
		:center(_center), scale(_scale) { build(); }

	void build() {
		vtx.clear();

		vtx.push_back(vec3(1, 0, 0));
		vtx.push_back(vec3(0, -1, 0));
		vtx.push_back(vec3(-1, 0, 0));
		vtx.push_back(vec3(0, 1, 0));
		vtx.push_back(vec3(0, 0, 1));
		vtx.push_back(vec3(0, 0, -1));

		for (auto& v : vtx) v = normalize(v) * scale + center;
	}

	std::vector<Intersectable*> create(std::vector<Intersectable*> objects) {
		objects.push_back(new Triangle(vtx[1], vtx[0], vtx[4]));
		objects.push_back(new Triangle(vtx[2], vtx[1], vtx[4]));
		objects.push_back(new Triangle(vtx[3], vtx[2], vtx[4]));
		objects.push_back(new Triangle(vtx[0], vtx[3], vtx[4]));
		objects.push_back(new Triangle(vtx[0], vtx[1], vtx[5]));
		objects.push_back(new Triangle(vtx[1], vtx[2], vtx[5]));
		objects.push_back(new Triangle(vtx[2], vtx[3], vtx[5]));
		objects.push_back(new Triangle(vtx[3], vtx[0], vtx[5]));
		return objects;
	}

	void moveTo(vec3 _center) {
		vec3 oldCenter = center;
		center = _center;
		vec3 newCenter = oldCenter - center;

		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x - newCenter.x, oldv.y - newCenter.y, oldv.z - newCenter.z);
		}
	}

	void rotateY(float angle) {
		angle = angle * (M_PI / 180.0f);
		vec3 currentCenter = center;
		moveTo(vec3(0.0f, 0.0f, 0.0f));
		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x * cosf(angle) - oldv.z * sinf(angle) + currentCenter.x,
				oldv.y + currentCenter.y,
				oldv.z * cosf(angle) + oldv.x * sinf(angle) + currentCenter.z);
		}
	}
};

//Source: https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html
struct Icosahedron {
	std::vector<vec3> vtx;
	vec3 center = vec3(0.0f, 0.0f, 0.0f);
	float scale = 0.2f;

	Icosahedron() { build(); }

	Icosahedron(vec3 _center, float _scale)
		:center(_center), scale(_scale) { build(); }

	void build() {
		vtx.clear();

		vtx.push_back(vec3(0, -0.525731, 0.850651));
		vtx.push_back(vec3(0.850651, 0, 0.525731));
		vtx.push_back(vec3(0.850651, 0, -0.525731));
		vtx.push_back(vec3(-0.850651, 0, -0.525731));
		vtx.push_back(vec3(-0.850651, 0, 0.525731));
		vtx.push_back(vec3(-0.525731, 0.850651, 0));
		vtx.push_back(vec3(0.525731, 0.850651, 0));
		vtx.push_back(vec3(0.525731, -0.850651, 0));
		vtx.push_back(vec3(-0.525731, -0.850651, 0));
		vtx.push_back(vec3(0, -0.525731, -0.850651));
		vtx.push_back(vec3(0, 0.525731, -0.850651));
		vtx.push_back(vec3(0, 0.525731, 0.850651));

		for (auto& v : vtx) v = normalize(v) * scale + center;
	}

	std::vector<Intersectable*> create(std::vector<Intersectable*> objects) {
		objects.push_back(new Triangle(vtx[1],  vtx[2],  vtx[6]) );
		objects.push_back(new Triangle(vtx[1],  vtx[7],  vtx[2]) );
		objects.push_back(new Triangle(vtx[3],  vtx[4],  vtx[5]) );
		objects.push_back(new Triangle(vtx[4],  vtx[3],  vtx[8]) );
		objects.push_back(new Triangle(vtx[6],  vtx[5],  vtx[11]));
		objects.push_back(new Triangle(vtx[5],  vtx[6],  vtx[10]));
		objects.push_back(new Triangle(vtx[9],  vtx[10], vtx[2]) );
		objects.push_back(new Triangle(vtx[10], vtx[9],  vtx[3]) );
		objects.push_back(new Triangle(vtx[7],  vtx[8],  vtx[9]) );
		objects.push_back(new Triangle(vtx[8],  vtx[7],  vtx[0]) );
		objects.push_back(new Triangle(vtx[11], vtx[0],  vtx[1]) );
		objects.push_back(new Triangle(vtx[0],  vtx[11], vtx[4]) );
		objects.push_back(new Triangle(vtx[6],  vtx[2],  vtx[10]));
		objects.push_back(new Triangle(vtx[1],  vtx[6],  vtx[11]));
		objects.push_back(new Triangle(vtx[3],  vtx[5],  vtx[10]));
		objects.push_back(new Triangle(vtx[5],  vtx[4],  vtx[11]));
		objects.push_back(new Triangle(vtx[2],  vtx[7],  vtx[9]) );
		objects.push_back(new Triangle(vtx[7],  vtx[1],  vtx[0]) );
		objects.push_back(new Triangle(vtx[3],  vtx[9],  vtx[8]) );
		objects.push_back(new Triangle(vtx[4],  vtx[8],  vtx[0]) );
		return objects;
	}

	void moveTo(vec3 _center) {
		vec3 oldCenter = center;
		center = _center;
		vec3 newCenter = oldCenter - center;

		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x - newCenter.x, oldv.y - newCenter.y, oldv.z - newCenter.z);
		}
	}

	void rotateY(float angle) {
		angle = angle * (M_PI / 180.0f);
		vec3 currentCenter = center;
		moveTo(vec3(0.0f, 0.0f, 0.0f));
		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x * cosf(angle) - oldv.z * sinf(angle) + currentCenter.x,
				oldv.y + currentCenter.y,
				oldv.z * cosf(angle) + oldv.x * sinf(angle) + currentCenter.z);
		}
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

// A globális illumináció példaprogramból származó pont-fényforrás objektum
// https://edu.vik.bme.hu/mod/url/view.php?id=97494
struct Light {
	vec3 location;
	vec3 power;

	Light(vec3 _location, vec3 _power) {
		location = _location;
		power = _power;
	}
	double distanceOf(vec3 point) {
		return length(location - point);
	}
	vec3 directionOf(vec3 point) {
		return normalize(location - point);
	}
	vec3 radianceAt(vec3 point) {
		double distance2 = dot(location - point, location - point);
		if (distance2 < epsilon) distance2 = epsilon;
		return power / distance2;
	}
};

class Scene {
public:
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;

	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		Cube cube = Cube(vec3(0.0f, 0.0f, 0.0f), 0.6f);
		cube.rotateY(45.0f);
		objects = cube.create(objects);
		
		Icosahedron i1 = Icosahedron(vec3(0.3f, -0.3f, 0.2f), 0.2f);
		objects = i1.create(objects);
		Octahedron o1 = Octahedron(vec3(-0.3f, -0.3f, 0.0f), 0.2f);
		objects = o1.create(objects);
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	void clear() {
		objects.clear();
		lights.clear();
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	int counter = 0;
	bool printing = false;

	bool count = false;

	vec3 trace(Ray ray) {
		if (count && counter++ > 20000) {
			printing = true;
			counter = 0;
		}
		Hit hit = firstIntersect(ray);	// Find visible surface
		if (hit.t < 0) return vec3(0, 0, 0);	// If there is no intersection
		float La = 0.2f * (1 + dot(hit.normal, -ray.dir));

		vec3 outRad = vec3(La, La, La);
		if (printing) printf("Base outRad: %f %f %f\n", outRad.x, outRad.y, outRad.z);
		for (auto light : lights) {	// Direct light source computation
			vec3 outDir = light->directionOf(hit.position);
			if (printing) printf("outDir: %f %f %f\n", outDir.x, outDir.y, outDir.z);
			Hit shadowHit = firstIntersect(Ray(hit.position + hit.normal * epsilon, outDir));
			if (shadowHit.t < epsilon || shadowHit.t > light->distanceOf(hit.position)) {	// if not in shadow
				double cosThetaL = dot(hit.normal, outDir);
				if (printing) printf("cosThetaL: %f\n", cosThetaL);
				if (cosThetaL >= epsilon) {
					outRad = outRad + light->power * cosThetaL * light->radianceAt(hit.position);
				}
			}
		}
		if (printing) printf("Final outRad: %f %f %f\n\n", outRad.x, outRad.y, outRad.z);
		printing = false;
		return outRad;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
std::vector<vec4> image(windowWidth * windowHeight);

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.clear();
	scene.build();
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

bool enabled = false;

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

int coneNumber = 0;
std::vector<vec3> conePositions, coneNormals;

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		printf("pX: %d, pY: %d\n", pX, pY);
		vec3 color = scene.trace(scene.camera.getRay(pX, windowHeight - pY));
		printf("color: %f %f %f\n\n", color.x, color.y, color.z);

		vec3 cP = scene.firstIntersect(scene.camera.getRay(pX, windowHeight - pY)).position;
		vec3 cN = scene.firstIntersect(scene.camera.getRay(pX, windowHeight - pY)).normal;
		printf("conePosition: %f %f %f\n", cP.x, cP.y, cP.z);
		printf("coneNormal: %f %f %f\n", cN.x, cN.y, cN.z);
		if (abs(length(cN)-1) < epsilon) {
			switch (coneNumber) {
			case 0:
				coneNumber++;
				scene.clear();
				conePositions.push_back(cP);
				coneNormals.push_back(cN);
				scene.objects.push_back(new Cone(conePositions[0], 0.1f, coneNormals[0], 25.0f));
				scene.lights.push_back(new Light(conePositions[0] + coneNormals[0] * 0.01f, vec3(0.9f, 0.1f, 0.1f)));
				break;
			case 1:
				coneNumber++;
				scene.clear();
				conePositions.push_back(cP);
				coneNormals.push_back(cN);
				scene.objects.push_back(new Cone(conePositions[0], 0.1f, coneNormals[0], 25.0f));
				scene.lights.push_back(new Light(conePositions[0] + coneNormals[0] * 0.01f, vec3(0.9f, 0.1f, 0.1f)));
				scene.objects.push_back(new Cone(conePositions[1], 0.1f, coneNormals[1], 25.0f));
				scene.lights.push_back(new Light(conePositions[1] + coneNormals[1] * 0.01f, vec3(0.1f, 0.9f, 0.1f)));
				break;
			case 2:
				coneNumber++;
				scene.clear();
				conePositions.push_back(cP);
				coneNormals.push_back(cN);
				scene.objects.push_back(new Cone(conePositions[0], 0.1f, coneNormals[0], 25.0f));
				scene.lights.push_back(new Light(conePositions[0] + coneNormals[0] * 0.01f, vec3(0.9f, 0.1f, 0.1f)));
				scene.objects.push_back(new Cone(conePositions[1], 0.1f, coneNormals[1], 25.0f));
				scene.lights.push_back(new Light(conePositions[1] + coneNormals[1] * 0.01f, vec3(0.1f, 0.9f, 0.1f)));
				scene.objects.push_back(new Cone(conePositions[2], 0.1f, coneNormals[2], 25.0f));
				scene.lights.push_back(new Light(conePositions[2] + coneNormals[2] * 0.01f, vec3(0.1f, 0.1f, 0.9f)));
				break;
			case 3:
				coneNumber = 0;
				scene.clear();
				conePositions.clear();
				coneNormals.clear();
				break;
			}
			
			scene.build();
			scene.render(image);


			fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
			glutPostRedisplay();
		}
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}