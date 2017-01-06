//Parallel Raytracer
//Use provided makefile to compile

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "cairo.h"
// #include "/usr/local/include/cairo/cairo.h"
// usr/local/Cellar/cairo/1.14.2_1

#include "vec3.h"

#define HEIGHT 512
#define WIDTH 512
#define PI 3.14159f

/* Negative distance values are obviously not a hit
   Sometimes explicitly set to -1.0f to express no hit*/

typedef struct Camera{
  vec3 point;
  float fieldOfView;
  vec3 toPoint;
  vec3 up;
} Camera;

typedef enum {OMNI, SPOT} lightType;

typedef struct Light{
  lightType type;
  vec3 point;
  vec3 color;
} Light;

typedef struct Sphere{
  vec3 point;
  float radius;
} Sphere;

typedef struct Triangle{
  vec3 point1;
  vec3 point2;
  vec3 point3;
} Triangle;

typedef enum {PHONG, ORIGINAL} materialType;

typedef struct Material{
  vec3 color;
  materialType type;
  int metal;
  float specular;
  float lambert;
  float ambient;
  float exponent;
} Material;

typedef enum {SPHERE, TRIANGLE} objectType;

typedef struct Object{
  int matIndex;
  objectType type;
  void* object;
} Object;

typedef struct Ray{
  vec3 point;
  vec3 vector;
} Ray;


typedef struct Scene{
  Camera* camera;
  Material* materials;
  Object* objects;
  Light* lights;
  int n_lights;
  int n_objects;
  int n_materials;
} Scene;

typedef struct Dist{
  float distance;
  Object* object;
} Dist;

__device__ vec3 objectNormal(Object* object, vec3 point);
__device__ float sphereIntersection(Sphere* sphere, Ray* ray);
__device__ vec3 sphereNormal(Sphere* sphere, vec3 pos);
__device__ vec3 triNormal(Triangle* tri);
__device__ float triIntersection(Triangle* tri, Ray* ray);
__device__ vec3 trace(Ray* ray, Scene* scene, int depth);
__device__ vec3 surfaceTrace(Ray* ray, Scene* scene, Object* object, vec3 pointAtTime, vec3 normal, float depth);
__device__  Dist intersectScene(Ray* ray, Scene* scene);
__device__  int isLightVisible(vec3 point, Scene* scene, vec3 light);
// __global__ void render(float* img); 
__host__ void tone_map(float* img, int size);
__device__ void printScene(Scene* scene);


__device__ vec3 objectNormal(Object* object, vec3 point){
  if (object->type == SPHERE){
    return sphereNormal((Sphere *) object->object, point);
  }

  if (object->type == TRIANGLE) {
    return (triNormal((Triangle *) object->object));
  }
  return ZERO;
}

__device__ float sphereIntersection(Sphere* sphere, Ray* ray){

  vec3 eye_to_center = subtract(sphere->point, ray->point);

  float v = dotProduct(eye_to_center, ray->vector);
  float eoDot = dotProduct(eye_to_center, eye_to_center);
  float discriminant = (sphere->radius * sphere->radius) - eoDot + (v * v);

  if ((fabsf(length(eye_to_center)-sphere->radius) / sphere->radius) < .001f) {
    return -1.0f;
  }

  // If the discriminant is negative, that means that the sphere hasn't
  // been hit by the ray
  if (discriminant < 0.0f) {
      return -1.0f;
  } else {
      // otherwise, we return the distance from the camera point to the sphere
      // `Math.sqrt(dotProduct(a, a))` is the length of a vector, so
      // `v - Math.sqrt(discriminant)` means the length of the the vector
      // just from the camera to the intersection point.
      return v - sqrtf(discriminant);
  }
}

__device__ vec3 sphereNormal(Sphere* sphere, vec3 pos){
  return unitVector(subtract(pos, sphere->point));
}

__device__ vec3 triNormal(Triangle* tri) {
    return unitVector(
        crossProduct(
          subtract(tri->point2, tri->point1),
          subtract(tri->point3, tri->point1))
         );
}

__device__ float triIntersection(Triangle* tri, Ray* ray) {
    // compute triangle normal and d in plane equation
    vec3 triNorm = triNormal(tri);
    float d = -1.0f * dotProduct(tri->point1, triNorm);

    // compute where ray intersects plane
    float dist = -1.0f * (dotProduct(ray->point, triNorm) + d) / dotProduct(ray->vector, triNorm);

    // if behind the ray starting point, we are done -- no intersection
    if (dist < 0.001f) {
      return -1.0f;
    }

    vec3 P = add(ray->point, scale(ray->vector, dist));

    // do inside test, edge by edge on triangle
    vec3 v1 = subtract(tri->point1, ray->point);
    vec3 v2 = subtract(tri->point2, ray->point);

    vec3 n1 = unitVector(crossProduct(v2, v1));
    float d1 = -1.0f * dotProduct(ray->point, n1);
    if((d1 + dotProduct(P, n1)) < 0.0f) {
      return -1.0f;
    }

    vec3 v3 = subtract(tri->point3, ray->point);
    n1 = unitVector(crossProduct(v3, v2));
    d1 = -1.0f * dotProduct(ray->point, n1);
    if((d1 + dotProduct(P, n1)) < 0.0f) {
      return -1.0f;
    }

    n1 = unitVector(crossProduct(v1, v3));
    d1 = -1.0f * dotProduct(ray->point, n1);
    if((d1 + dotProduct(P, n1)) < 0.0f) {
      return -1.0f;
    }

    return dist;
}


__device__ vec3 surfaceTrace(Ray* ray, Scene* scene, Object* object, vec3 pointAtTime, vec3 normal, float depth) {

  Material* material = &(scene->materials[object->matIndex]);
  vec3 objColor = material->color;
  vec3 c = ZERO;
  // vec3 specReflect = ZERO;
  vec3 lambertAmount = ZERO;

  // lambert shading
  if (material->lambert > 0.0f) {
    for (int i = 0; i < scene->n_lights; i++) {
      vec3 lightPoint = scene->lights[i].point;

      if (isLightVisible(pointAtTime, scene, lightPoint)){
        // lambertian reflectance
        float contribution = dotProduct(unitVector(subtract(lightPoint, pointAtTime)), normal);

        if(contribution > 0.0f) {
          lambertAmount = add(lambertAmount, scale(scene->lights[i].color, contribution));
        }

      }
    }
  }
    // for assn 6, adjust lit color by object color and divide by 255 since light color is 0 to 255
  lambertAmount = compScale(lambertAmount, objColor);
  lambertAmount = scale(lambertAmount, material->lambert);
  lambertAmount = scale(lambertAmount, 1./255.);

  // specular
  if(material->specular > 0.0f) {

    Ray reflectedRay = {
        .point  = pointAtTime,
        .vector = reflectThrough(scale(ray->vector, -1.0f), normal)
    };

    vec3 reflectedColor = trace(&reflectedRay, scene, depth+1);

    if (material->type == PHONG){

        for (int i = 0; i < scene->n_lights; i++) {
            vec3 lightPoint = scene->lights[i].point;

            if (isLightVisible(pointAtTime, scene, lightPoint)) {

                vec3 H = unitVector(add(
                            scale(unitVector(ray->vector), -1.0f),
                            unitVector(subtract(scene->lights[i].point, pointAtTime))));

                float ndoth = dotProduct(normal, H);

                if(ndoth < 0.0f) {
                  ndoth = 0.0f;
                }

                float intensity = powf(ndoth, material->exponent);

                vec3 addColor;

                if(!material->metal){
                    addColor = scale(scene->lights[i].color, intensity);
                }else{
                    addColor = scale(objColor, intensity);
                }

                c = add(c, addColor);
            }
        }
    }

    c = add(c, reflectedColor);
    c = scale(c, material->specular);
  }

  return add3(c, lambertAmount, scale(objColor, material->ambient));
}


__device__ vec3 trace(Ray* ray, Scene* scene, int depth) {

  // printScene(scene);

  if (depth > 3) {
    return ZERO; //--Need ret val---///
  }

  Dist distObject = intersectScene(ray, scene);

  // If we don't hit anything, fill this pixel with the background color -
  // in this case, white.
  if (distObject.distance < 0.0f) {
      return ZERO;
  }

  float dist = distObject.distance;
  Object* object = distObject.object;

  // The `pointAtTime` is another way of saying the 'intersection point'
  // of this ray into this object. We compute this by simply taking
  // the direction of the ray and making it as long as the distance
  // returned by the intersection check.
  vec3 pointAtTime = add(ray->point, scale(ray->vector, dist));

  // for Assn 6, generalize to objectNormal
  return surfaceTrace(ray, scene, object, pointAtTime, objectNormal(object, pointAtTime), depth);
}


__device__ Dist intersectScene(Ray* ray, Scene* scene) {

  Dist closest = { .distance = FLT_MAX, .object = NULL};

  // Find closest intersecting object
  for (int i = 0; i < scene->n_objects; i++) {

    Object* object = &(scene->objects[i]);

    float dist = -1.0f;

    if (object->type == SPHERE)  {
      dist = sphereIntersection((Sphere *) object->object, ray);
    }

    if (object->type == TRIANGLE) {
      dist = triIntersection ((Triangle *) object->object, ray);
    }

    if (dist > .0f && dist < closest.distance) {
      closest.distance = dist;
      closest.object = object;
    }
  }

  if (closest.distance == FLT_MAX) {
    closest.distance = -1.0f;
  }

  return closest;
}

__device__ int isLightVisible(vec3 point, Scene* scene, vec3 light) {

  vec3 vector = unitVector(subtract(light, point));
  Ray ray = { .point = point, .vector = vector};

  Dist distObject =  intersectScene(&ray, scene);

  if (distObject.distance > 0.0f && distObject.distance < (length(vector) -.005)) {
    return 0; // False case where an object is found between the point and the light
  }
  else {
    return 1; // Either negative (no collision at all) or >length(vector) (light is closer than collision)
  }
}


__device__ void printScene(Scene* scene){

  Camera* camera = scene->camera;
  Material* materials = scene->materials;
  Object* objects = scene->objects;
  Light* lights = scene->lights;
  int n_materials = scene->n_materials;
  int n_objects = scene->n_objects;
  int n_lights = scene->n_lights;

  printf("Camera of scene is at point (%f, %f, %f)\n", camera->point.x, camera->point.y, camera->point.z);
  printf("Camera FOV is %f\n", camera->fieldOfView);
  printf("Camera toPoint is (%f, %f, %f)\n", camera->toPoint.x, camera->toPoint.y, camera->toPoint.z);
  printf("Camera up is (%f, %f, %f)\n", camera->up.x, camera->up.y, camera->up.z);

  //Check materials
  for(int i=0;i<n_materials;i++){
    printf("Materials[%d] has color=(%f, %f, %f), type=%d, metal=%d,\n", i, materials[i].color.x, materials[i].color.y, materials[i].color.z, materials[i].type, materials[i].metal);
    printf("\t specular=%f, lambert=%f, ambient=%f, exponent=%f\n", materials[i].specular, materials[i].lambert, materials[i].ambient, materials[i].exponent);
  }

  //Check Objects
  for(int i=0;i<n_objects;i++){
    if(objects[i].type == SPHERE){
      printf("objects[%d] is SPHERE with matIndex=%d\n", i, objects[i].matIndex);
      Sphere* sph = (Sphere*) objects[i].object;
      printf("\tSphere point=(%f, %f, %f), radius=%f\n", sph->point.x, sph->point.y, sph->point.z, sph->radius);
    }
    if(objects[i].type == TRIANGLE){
      printf("objects[%d] is TRIANGLE with matIndex=%d\n", i, objects[i].matIndex);
      Triangle* tri = (Triangle*) objects[i].object;
      printf("\tTriangle point1=(%f, %f, %f), point2=(%f, %f, %f), point3=(%f, %f, %f)\n", tri->point1.x, tri->point1.y, tri->point1.z, tri->point2.x, tri->point2.y, tri->point2.z, tri->point3.x, tri->point3.y, tri->point3.z);
    }
  }

  //Check Lights
  for(int i=0;i<n_lights;i++){
    printf("Lights[%d] has type=%d, point=(%f, %f, %f), color=(%f, %f, %f)\n", i, lights[i].type, lights[i].point.x, lights[i].point.y, lights[i].point.z, lights[i].color.x, lights[i].color.y, lights[i].color.z);
  }

}

__device__ void render(float* img, Scene* scene) {


  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = tx + blockDim.x * blockIdx.x;
  int row = ty + blockDim.y * blockIdx.y;

  if(row<HEIGHT && col<WIDTH){

    Camera* camera = scene->camera;

    // // first thread checks its scene contents
    // if(row==0 && col==0){
    //   printScene(scene);
    // }

    vec3 eyeVector = unitVector(subtract(camera->toPoint, camera->point));
    vec3 vpRight = unitVector(crossProduct(eyeVector, camera->up));
    vec3 vpUp = unitVector(crossProduct(vpRight, eyeVector));

    float height = (float) HEIGHT;
    float width = (float) WIDTH;

    float fovRadians = PI * (camera->fieldOfView / 2.0f) / 180.0f;
    float heightWidthRatio = height / width;
    float halfWidth = tanf(fovRadians);
    float halfHeight = heightWidthRatio * halfWidth;
    float camerawidth = halfWidth * 2.0f;
    float cameraheight = halfHeight * 2.0f;
    float pixelWidth = camerawidth / (width - 1.0f);
    float pixelHeight = cameraheight / (height - 1.0f);

    Ray ray;
    ray.point = camera->point;

    int x= col;
    int y= row;

    vec3 color = ZERO;

    //antialiasing with 9 samples
    for (float s = -.4f; s < .6f; s+=.3f) {
      for (float r = -.4f; r < .6; r +=.3f) {

        vec3 xcomp = scale(vpRight, ((x+s) * pixelWidth) - halfWidth);
        vec3 ycomp = scale(vpUp, ((y+r) * pixelHeight) - halfHeight);

        ray.vector = unitVector(add3(eyeVector, xcomp, ycomp));

        // use the vector generated to raytrace the scene, returning a color
        // as a `{x, y, z}` vector of RGB values
        color = add(color, trace(&ray, scene, 0));
      }
    }

    int index = (x * 4) + (y * width * 4);
    img[index + 0] = 0.0f;
    img[index + 1] = color.x;
    img[index + 2] = color.y;
    img[index + 3] = color.z;
  }
}


typedef struct Pixel{
  unsigned char A;
  unsigned char R;
  unsigned char G;
  unsigned char B;
} Pixel;

__host__ void tone_map(float* img, int size){

  float max = img[0];

  for (int i=0; i<size; i++){
    if (img[i] > max){
      max = img[i];
    }
  }

  for (int i=0; i<size; i++){
    img[i] = img[i] / max * 255.0f;
  }

}

__global__ void callRender(float* img){

  Camera s_camera = {
      .point = {
          .x = 50.0f,
          .y = 50.0f,
          .z = 400.0
      },
      .fieldOfView = 40.0f,
      .toPoint = {
          .x = 50.0f,
          .y = 50.0f,
          .z = 0.0f
      },
      .up = UP
  };

  ////////////////////Lights////////////////
  Light s_lights[3] = {
    {
      .type = OMNI,
      .point = {
        .x = 50.0f,
        .y = 95.0f,
        .z = 50.0f
      },
      .color = {
        .x = 155.0f,
        .y = 155.0f,
        .z = 155.0f
      },
    },

    {
      .type = OMNI,
      .point = {
        .x = 5.0f,
        .y = 95.0f,
        .z = 100.0f
      },
      .color = {
        .x = 255.0f,
        .y = 220.0f,
        .z = 200.0f
      },
    },

    {
      .type = OMNI,
      .point = {
        .x = 95.0f,
        .y = 5.0f,
        .z = 100.0f
      },
      .color = {
        .x = 50.0f,
        .y = 50.0f,
        .z = 100.0f
      },
    }
  };

  Sphere s_spheres[2] = {

    {
      .point =
        {
          .x = 70.0f,
          .y = 25.0f,
          .z = 50.0f
        },
      .radius = 25.0f
    },

    {
      .point =
        {
          .x = 20.0f,
          .y = 10.0f,
          .z = 50.0f
        },
      .radius = 10.0f
    },

  };

  Triangle s_triangles[10] = {

    //Back wall
    {
      .point1 =
        {
          .x = 0.0f,
          .y = 0.0f,
          .z = 0.0f,
        },
      .point2 =
        {
          .x = 100.0f,
          .y = 0.0f,
          .z = 0.0f,
        },
      .point3 =
        {
          .x = 0.0f,
          .y = 100.0f,
          .z = 0.0f,
        }
    },

    {
      .point1 =
        {
          .x = 100.0f,
          .y = 0.0f,
          .z = 0.0f,
        },
      .point2 =
        {
          .x = 100.0f,
          .y = 100.0f,
          .z = 0.0f,
        },
      .point3 =
        {
          .x = 0.0f,
          .y = 100.0f,
          .z = 0.0f,
        }
    },

    //FLoor

    {
      .point1 =
        {
          .x = 0.0f,
          .y = 0.0f,
          .z = 0.0f,
        },
      .point2 =
        {
          .x = 0.0f,
          .y = 0.0f,
          .z = 100.0f,
        },
      .point3 =
        {
          .x = 100.0f,
          .y = 0.0f,
          .z = 100.0f,
        }
    },

    {
      .point1 =
        {
          .x = 100.0f,
          .y = 0.0f,
          .z = 100.0f,
        },
      .point2 =
        {
          .x = 100.0f,
          .y = 0.0f,
          .z = 0.0f,
        },
      .point3 =
        {
          .x = 0.0f,
          .y = 0.0f,
          .z = 0.0f,
        }
    },

    //Ceiling

    {
      .point1 =
        {
          .x = 0.0f,
          .y = 100.0f,
          .z = 0.0f,
        },
      .point2 =
        {
          .x = 100.0f,
          .y = 100.0f,
          .z = 0.0f,
        },
      .point3 =
        {
          .x = 100.0f,
          .y = 100.0f,
          .z = 100.0f,
        }
    },

    {
      .point1 =
        {
          .x = 100.0f,
          .y = 100.0f,
          .z = 100.0f,
        },
      .point2 =
        {
          .x = 0.0f,
          .y = 100.0f,
          .z = 100.0f,
        },
      .point3 =
        {
          .x = 0.0f,
          .y = 100.0f,
          .z = 0.0f,
        }
    },

    //Left wall red

    {
      .point1 =
        {
          .x = 0.0f,
          .y = 0.0f,
          .z = 0.0f,
        },
      .point2 =
        {
          .x = 0.0f,
          .y = 100.0f,
          .z = 0.0f,
        },
      .point3 =
        {
          .x = 0.0f,
          .y = 100.0f,
          .z = 100.0f,
        }
    },

    {
      .point1 =
        {
          .x = 0.0f,
          .y = 100.0f,
          .z = 100.0f,
        },
      .point2 =
        {
          .x = 0.0f,
          .y = 0.0f,
          .z = 100.0f,
        },
      .point3 =
        {
          .x = 0.0f,
          .y = 0.0f,
          .z = 0.0f,
        }
    },

    // RIght Wall blue
    {
      .point1 =
        {
          .x = 100.0f,
          .y = 0.0f,
          .z = 0.0f,
        },
      .point2 =
        {
          .x = 100.0f,
          .y = 0.0f,
          .z = 100.0f,
        },
      .point3 =
        {
          .x = 100.0f,
          .y = 100.0f,
          .z = 100.0f,
        }
    },

    {
      .point1 =
        {
          .x = 100.0f,
          .y = 100.0f,
          .z = 100.0f,
        },
      .point2 =
        {
          .x = 100.0f,
          .y = 100.0f,
          .z = 0.0f,
        },
      .point3 =
        {
          .x = 100.0f,
          .y = 0.0f,
          .z = 0.0f,
        }
    }
  };

  Object s_objects[12] = {

    {
      .type = SPHERE,
      .matIndex = 4,
      .object = &(s_spheres[0])
    },

    {
      .type = SPHERE,
      .matIndex = 5,
      .object = &(s_spheres[1])
    },

    // Back wall
    {
      .type = TRIANGLE,
      .matIndex = 1,
      .object = &(s_triangles[0])
    },

    {
      .type = TRIANGLE,
      .matIndex = 1,
      .object = &(s_triangles[1])
    },

    // FLoor
    {
      .type = TRIANGLE,
      .matIndex = 1,
      .object = &(s_triangles[2])
    },

    {
      .type = TRIANGLE,
      .matIndex = 1,
      .object = &(s_triangles[3])
    },

    // Ceiling
    {
      .type = TRIANGLE,
      .matIndex = 1,
      .object = &(s_triangles[4])
    },

    {
      .type = TRIANGLE,
      .matIndex = 1,
      .object = &(s_triangles[5])
    },

    // Left Wall Red
    {
      .type = TRIANGLE,
      .matIndex = 2,
      .object = &(s_triangles[6])
    },

    {
      .type = TRIANGLE,
      .matIndex = 2,
      .object = &(s_triangles[7])
    },

    // Right wall blue
    {
      .type = TRIANGLE,
      .matIndex = 3,
      .object = &(s_triangles[8])
    },

    {
      .type = TRIANGLE,
      .matIndex = 3,
      .object = &(s_triangles[9])
    }
  };

  Material s_materials[6] = {

    // Mat 0
    {
      .color =
        {
          .x = 255.0f,
          .y = 255.0f,
          .z = 255.0f,
        },
      .type = ORIGINAL,
      .metal = 0,
      .specular = 0.0f,
      .lambert = 0.85f,
      .ambient = 0.05f,
      .exponent = 0.0f,
    },

    // Mat 1 Diff white
    {
      .color =
        {
          .x = 255.0f,
          .y = 255.0f,
          .z = 255.0f,
        },
      .type = ORIGINAL,
      .metal = 0,
      .specular = 0.0f,
      .lambert = 0.9f,
      .ambient = 0.05f,
      .exponent = 0.0f,
    },

    // Mat 2 Diff Red
    {
      .color =
        {
          .x = 255.0f,
          .y = 90.0f,
          .z = 90.0f,
        },
      .type = ORIGINAL,
      .metal = 0,
      .specular = 0.0f,
      .lambert = 0.9f,
      .ambient = 0.1f,
      .exponent = 0.0f,
    },

    // Mat 3 Diffuse Blue
    {
      .color =
        {
          .x = 90.0f,
          .y = 90.0f,
          .z = 255.0f,
        },
      .type = ORIGINAL,
      .metal = 0,
      .specular = 0.0f,
      .lambert = 0.9f,
      .ambient = 0.1f,
      .exponent = 0.0f,
    },

    // Mat 4 Mirror
    {
      .color =
        {
          .x = 255.0f,
          .y = 255.0f,
          .z = 255.0f,
        },
      .type = ORIGINAL,
      .metal = 0,
      .specular = 0.9f,
      .lambert = 0.1f,
      .ambient = 0.0f,
      .exponent = 0.0f,
    },

    // Mat 5 GOLD!
    {
      .color =
        {
          .x = 200.0f,
          .y = 170.0f,
          .z = 60.0f,
        },
      .type = PHONG,
      .metal = 1,
      .specular = 0.5f,
      .lambert = 0.4f,
      .ambient = 0.1f,
      .exponent = 2.0f,
    }
  };

  Scene s_scene = {
    .camera = &s_camera,
    .materials = s_materials,
    .objects = s_objects,
    .lights = s_lights,
    .n_lights = 3,
    .n_materials = 6,
    .n_objects = 12
  };


  // int tx = threadIdx.x;
  // int ty = threadIdx.y;
  // int col = tx + blockDim.x * blockIdx.x;
  // int row = ty + blockDim.y * blockIdx.y;

  render(img, &s_scene);
}

int main (){

  //////////////////VARIABLES//////////////////
  int gpucount = 0; // Count of available GPUs
  int gpunum = 0; // Device number to use
  int Block_Dim = 32;
  int Grid_Dim_x = WIDTH/Block_Dim;  //512/32 = 16
  int Grid_Dim_y = HEIGHT/Block_Dim;

  dim3 Grid(Grid_Dim_x, Grid_Dim_y);
  dim3 Block(Block_Dim, Block_Dim);

  cudaEvent_t start, stop; // using cuda events to measure time
  float elapsed_time_ms; // which is applicable for asynchronous code also
  cudaError_t errorcode;

  ////////////////////SET UP CUDA/////////////////////////////////
  errorcode = cudaGetDeviceCount(&gpucount);
  if (errorcode == cudaErrorNoDevice) {
    printf("No GPUs are visible\n");
    exit(-1);
  }
  else {
     printf("Device count = %d\n",gpucount);
  }
  cudaSetDevice(gpunum);
  printf("Using device %d\n",gpunum);


  ////////////////////CUDA/////////////////////////////////////

  cudaDeviceSetLimit(cudaLimitStackSize, 50*1024);
  //105090

  float* dev_img;
  cudaMalloc((void**)&dev_img, sizeof(float)*HEIGHT*WIDTH*4);

  cudaEventCreate(&start); // instrument code to measure start time
  cudaEventCreate(&stop);
  
  cudaEventRecord(start, 0);

  callRender<<<Grid,Block>>>(dev_img);

  cudaEventRecord(stop, 0); // instrument code to measure end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  float*img = (float *) malloc(sizeof(float)*HEIGHT*WIDTH*4);
  cudaMemcpy(img,dev_img, sizeof(float)*HEIGHT*WIDTH*4, cudaMemcpyDeviceToHost);

  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); // exec. time


  /////////////////HOST CALCULATIONS//////////////////////////
  tone_map(img, HEIGHT * WIDTH * 4);

  Pixel* imgData = (Pixel *) malloc(sizeof(Pixel) * WIDTH * HEIGHT);

  for (int y=0; y<HEIGHT; y++){
    for (int x=0; x<WIDTH; x++){
      
      int indexOld = (y * WIDTH * 4) + (x * 4);
      int indexNew = ((HEIGHT - y - 1) * WIDTH) + x;
      imgData[indexNew].B = (unsigned char) img[indexOld + 0];
      imgData[indexNew].G = (unsigned char) img[indexOld + 1];
      imgData[indexNew].R = (unsigned char) img[indexOld + 2];
      imgData[indexNew].A = (unsigned char) img[indexOld + 3];

    }
  }

  int stride = cairo_format_stride_for_width (CAIRO_FORMAT_RGB24, WIDTH);

  cairo_surface_t* surface_cairo =
    cairo_image_surface_create_for_data ((unsigned char*)imgData, CAIRO_FORMAT_RGB24, WIDTH, HEIGHT, stride);

  cairo_surface_write_to_png (surface_cairo, "parallel.png");
  cairo_surface_destroy (surface_cairo);

  free(img);
  free(imgData);
  cudaFree(dev_img);

  return 0;
}
