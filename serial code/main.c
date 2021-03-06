//Serial Raytracer
//Compile command:
//gcc -std=c99 -o main -L/usr/local/lib/cairo/ -lcairo main.c -I/usr/local/include/cairo/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#include "cairo.h"
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


//Fill in SceneStruct later
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

vec3 objectNormal(Object* object, vec3 point);
float sphereIntersection(Sphere* sphere, Ray* ray);
vec3 sphereNormal(Sphere* sphere, vec3 pos);
vec3 triNormal(Triangle* tri);
float triIntersection(Triangle* tri, Ray* ray);
float* render(Scene* scene);
vec3 trace(Ray* ray, Scene* scene, int depth);
Dist intersectScene(Ray* ray, Scene* scene);
int isLightVisible(vec3 point, Scene* scene, vec3 light);
vec3 surface(Ray* ray, Scene* scene, Object* object, vec3 pointAtTime, vec3 normal, float depth);


//timer function
double get_time_ms()
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return (t.tv_sec + (t.tv_usec / 1000000.0)) * 1000.0;
}


vec3 objectNormal(Object* object, vec3 point){

  if (object->type == SPHERE){
    return sphereNormal(object->object, point);
  }

  if (object->type == TRIANGLE) {
    return (triNormal(object->object));
  }

  printf("objectNormal broke\n");
  return ZERO;

}

float sphereIntersection(Sphere* sphere, Ray* ray){

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

vec3 sphereNormal(Sphere* sphere, vec3 pos){
  return unitVector(subtract(pos, sphere->point));
}

vec3 triNormal(Triangle* tri) {
    return unitVector(
        crossProduct(
          subtract(tri->point2, tri->point1),
          subtract(tri->point3, tri->point1))
         );
}

float triIntersection(Triangle* tri, Ray* ray) {

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



float* render(Scene* scene) {

  float* img = malloc(sizeof(float) * HEIGHT * WIDTH * 4);

  Camera* camera = scene->camera;
  Object* objects = scene->objects;
  Light* lights = scene->lights;
  int n_objects = scene->n_objects;
  int n_lights = scene->n_objects;

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

  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {

        // Antialiasing with 9 samples
        vec3 color = ZERO;

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
  return img;
}


vec3 trace(Ray* ray, Scene* scene, int depth) {

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
        return surface(ray, scene, object, pointAtTime, objectNormal(object, pointAtTime), depth);
}

Dist intersectScene(Ray* ray, Scene* scene) {

  Dist closest = { .distance = FLT_MAX, .object = NULL};

  // Find closest intersecting object
  for (int i = 0; i < scene->n_objects; i++) {

    Object* object = &(scene->objects[i]);

    float dist = -1.0f;

    if (object->type == SPHERE)  {
      dist = sphereIntersection(object->object, ray);
    }

    if (object->type == TRIANGLE) {
      dist = triIntersection (object->object, ray);
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

int isLightVisible(vec3 point, Scene* scene, vec3 light) {

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

vec3 surface(Ray* ray, Scene* scene, Object* object, vec3 pointAtTime, vec3 normal, float depth) {

  Material* material = &(scene->materials[object->matIndex]);
  vec3 objColor = material->color;
  vec3 c = ZERO;
  vec3 specReflect = ZERO;
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

typedef struct Pixel{
  unsigned char A;
  unsigned char R;
  unsigned char G;
  unsigned char B;
} Pixel;

void tone_map(float* img, int size){

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

int main (){

  double start, end, elapsed;

  #include "scene.h"

  //Start time
  start = get_time_ms();

  // This is MALLOCED!!
  float* img = render(&s_scene);

  //End time 
  end = get_time_ms();
  elapsed = end - start;
  printf("Serial raytracer rendered image in %f milliseconds\n", elapsed);

  tone_map(img, HEIGHT * WIDTH * 4);

  // printf("Rendered! \n");

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


  cairo_surface_write_to_png (surface_cairo, "serial.png");
  cairo_surface_destroy (surface_cairo);

  free(img);
  free(imgData);

  return 0;
}
