//Create scene

// __device__ Scene createScene(){


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

  // printf("Creating scene, camera is at position (%f, %f, %f)\n", s_camera.point.x, s_camera.point.y, s_camera.point.z);
  // printf("Creating scene, camera FOV is %f\n", s_camera.fieldOfView);

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


//   return s_scene;
// }



// //malloc space
// __host__ Camera* createCamera(){

//   Camera* s_camera = (Camera *) malloc(sizeof(Camera) * 1);

//   // (s_camera).point.x = 50.0f;
//   // (s_camera).point.y = 50.0f;
//   // (s_camera).point.z = 400.0f;
//   // (s_camera).fieldOfView = 40.0f;
//   // (s_camera).toPoint.x = 50.0f;
//   // (s_camera).toPoint.y = 50.0f;
//   // (s_camera).toPoint.z = 0.0f;
//   // (s_camera).up.x = 0.0f;
//   // (s_camera).up.y = 1.0f;
//   // (s_camera).up.z = 0.0f;

//   s_camera[0].point.x = 50.0f;
//   s_camera[0].point.y = 50.0f;
//   s_camera[0].point.z = 400.0f;
//   s_camera[0].fieldOfView = 40.0f;
//   s_camera[0].toPoint.x = 50.0f;
//   s_camera[0].toPoint.y = 50.0f;
//   s_camera[0].toPoint.z = 0.0f;
//   s_camera[0].up.x = 0.0f;
//   s_camera[0].up.y = 1.0f;
//   s_camera[0].up.z = 0.0f;

//   return s_camera;
// }

// //malloc space
// __host__ Light* createLights(){
//   Light *s_lights = (Light *) malloc(sizeof(Light) * 3);

//   (s_lights[0]).type = OMNI;
//   (s_lights[0]).point.x = 50.0f;
//   (s_lights[0]).point.y = 95.0f;
//   (s_lights[0]).point.z = 50.0f;
//   (s_lights[0]).color.x = 155.0f;
//   (s_lights[0]).color.y = 155.0f;
//   (s_lights[0]).color.z = 155.0f;

//   (s_lights[1]).type = OMNI;
//   (s_lights[1]).point.x = 5.0f;
//   (s_lights[1]).point.y = 95.0f;
//   (s_lights[1]).point.z = 100.0f;
//   (s_lights[1]).color.x = 255.0f;
//   (s_lights[1]).color.y = 220.0f;
//   (s_lights[1]).color.z = 200.0f;

//   (s_lights[2]).type = OMNI;
//   (s_lights[2]).point.x = 95.0f;
//   (s_lights[2]).point.y = 5.0f;
//   (s_lights[2]).point.z = 100.0f;
//   (s_lights[2]).color.x = 50.0f;
//   (s_lights[2]).color.y = 50.0f;
//   (s_lights[2]).color.z = 100.0f;

//   return s_lights;
// }


// //malloc space 
// __host__ Sphere* createSpheres(){
//   Sphere* s_spheres = (Sphere *) malloc(sizeof(Sphere) * 2);

//   (s_spheres[0]).point.x = 70.0f;
//   (s_spheres[0]).point.y = 25.0f;
//   (s_spheres[0]).point.z = 50.0f;
//   (s_spheres[0]).radius = 25.0f;

//   (s_spheres[1]).point.x = 20.0f;
//   (s_spheres[1]).point.y = 10.0f;
//   (s_spheres[1]).point.z = 50.0f;
//   (s_spheres[1]).radius = 10.0f;

//   return s_spheres;

// }

// //malloc space
// __host__ Triangle* createTriangles(){
//   Triangle* s_triangles = (Triangle *) malloc(sizeof(Triangle) * 10);

//   //Back wall
//   (s_triangles[0]).point1.x = 0.0f;
//   (s_triangles[0]).point1.y = 0.0f;
//   (s_triangles[0]).point1.z = 0.0f;
//   (s_triangles[0]).point2.x = 100.0f;
//   (s_triangles[0]).point2.y = 0.0f;
//   (s_triangles[0]).point2.z = 0.0f;
//   (s_triangles[0]).point3.x = 0.0f;
//   (s_triangles[0]).point3.y = 100.0f;
//   (s_triangles[0]).point3.z = 0.0f;

//   (s_triangles[1]).point1.x = 100.0f;
//   (s_triangles[1]).point1.y = 0.0f;
//   (s_triangles[1]).point1.z = 0.0f;
//   (s_triangles[1]).point2.x = 100.0f;
//   (s_triangles[1]).point2.y = 100.0f;
//   (s_triangles[1]).point2.z = 0.0f;
//   (s_triangles[1]).point3.x = 0.0f;
//   (s_triangles[1]).point3.y = 100.0f;
//   (s_triangles[1]).point3.z = 0.0f;

//   //Floor
//   (s_triangles[2]).point1.x = 0.0f;
//   (s_triangles[2]).point1.y = 0.0f;
//   (s_triangles[2]).point1.z = 0.0f;
//   (s_triangles[2]).point2.x = 0.0f;
//   (s_triangles[2]).point2.y = 0.0f;
//   (s_triangles[2]).point2.z = 100.0f;
//   (s_triangles[2]).point3.x = 100.0f;
//   (s_triangles[2]).point3.y = 0.0f;
//   (s_triangles[2]).point3.z = 100.0f;

//   (s_triangles[3]).point1.x = 100.0f;
//   (s_triangles[3]).point1.y = 0.0f;
//   (s_triangles[3]).point1.z = 100.0f;
//   (s_triangles[3]).point2.x = 100.0f;
//   (s_triangles[3]).point2.y = 0.0f;
//   (s_triangles[3]).point2.z = 0.0f;
//   (s_triangles[3]).point3.x = 0.0f;
//   (s_triangles[3]).point3.y = 0.0f;
//   (s_triangles[3]).point3.z = 0.0f;

//   //Ceiling
//   (s_triangles[4]).point1.x = 0.0f;
//   (s_triangles[4]).point1.y = 100.0f;
//   (s_triangles[4]).point1.z = 0.0f;
//   (s_triangles[4]).point2.x = 100.0f;
//   (s_triangles[4]).point2.y = 100.0f;
//   (s_triangles[4]).point2.z = 0.0f;
//   (s_triangles[4]).point3.x = 100.0f;
//   (s_triangles[4]).point3.y = 100.0f;
//   (s_triangles[4]).point3.z = 100.0f;

//   (s_triangles[5]).point1.x = 100.0f;
//   (s_triangles[5]).point1.y = 100.0f;
//   (s_triangles[5]).point1.z = 100.0f;
//   (s_triangles[5]).point2.x = 0.0f;
//   (s_triangles[5]).point2.y = 100.0f;
//   (s_triangles[5]).point2.z = 100.0f;
//   (s_triangles[5]).point3.x = 0.0f;
//   (s_triangles[5]).point3.y = 100.0f;
//   (s_triangles[5]).point3.z = 0.0f;

//   //Left wall red
//   (s_triangles[6]).point1.x = 0.0f;
//   (s_triangles[6]).point1.y = 0.0f;
//   (s_triangles[6]).point1.z = 0.0f;
//   (s_triangles[6]).point2.x = 0.0f;
//   (s_triangles[6]).point2.y = 100.0f;
//   (s_triangles[6]).point2.z = 0.0f;
//   (s_triangles[6]).point3.x = 0.0f;
//   (s_triangles[6]).point3.y = 100.0f;
//   (s_triangles[6]).point3.z = 100.0f;

//   (s_triangles[7]).point1.x = 0.0f;
//   (s_triangles[7]).point1.y = 100.0f;
//   (s_triangles[7]).point1.z = 100.0f;
//   (s_triangles[7]).point2.x = 0.0f;
//   (s_triangles[7]).point2.y = 0.0f;
//   (s_triangles[7]).point2.z = 100.0f;
//   (s_triangles[7]).point3.x = 0.0f;
//   (s_triangles[7]).point3.y = 0.0f;
//   (s_triangles[7]).point3.z = 0.0f;

//   //Right wall blue
//   (s_triangles[8]).point1.x = 100.0f;
//   (s_triangles[8]).point1.y = 0.0f;
//   (s_triangles[8]).point1.z = 0.0f;
//   (s_triangles[8]).point2.x = 100.0f;
//   (s_triangles[8]).point2.y = 0.0f;
//   (s_triangles[8]).point2.z = 100.0f;
//   (s_triangles[8]).point3.x = 100.0f;
//   (s_triangles[8]).point3.y = 100.0f;
//   (s_triangles[8]).point3.z = 100.0f;

//   (s_triangles[9]).point1.x = 100.0f;
//   (s_triangles[9]).point1.y = 100.0f;
//   (s_triangles[9]).point1.z = 100.0f;
//   (s_triangles[9]).point2.x = 100.0f;
//   (s_triangles[9]).point2.y = 100.0f;
//   (s_triangles[9]).point2.z = 0.0f;
//   (s_triangles[9]).point3.x = 100.0f;
//   (s_triangles[9]).point3.y = 0.0f;
//   (s_triangles[9]).point3.z = 0.0f;

//   return s_triangles;

// }


// //malloc space
// __host__ Object* createObjects(Sphere* s_spheres, Triangle* s_triangles){
//   Object* s_objects = (Object *) malloc(sizeof(Object *) * 12);

//   (s_objects[0]).type = SPHERE;
//   (s_objects[0]).matIndex = 4;
//   (s_objects[0]).object = &s_spheres[0];

//   (s_objects[1]).type = SPHERE;
//   (s_objects[1]).matIndex = 5;
//   (s_objects[1]).object = &s_spheres[1];

//   //Back wall
//   (s_objects[2]).type = TRIANGLE;
//   (s_objects[2]).matIndex = 1;
//   (s_objects[2]).object = &s_triangles[0];

//   (s_objects[3]).type = TRIANGLE;
//   (s_objects[3]).matIndex = 1;
//   (s_objects[3]).object = &s_triangles[1];

//   //Floor
//   (s_objects[4]).type = TRIANGLE;
//   (s_objects[4]).matIndex = 1;
//   (s_objects[4]).object = &s_triangles[2];

//   (s_objects[5]).type = TRIANGLE;
//   (s_objects[5]).matIndex = 1;
//   (s_objects[5]).object = &s_triangles[3];

//   //Ceiling
//   (s_objects[6]).type = TRIANGLE;
//   (s_objects[6]).matIndex = 1;
//   (s_objects[6]).object = &s_triangles[4];

//   (s_objects[7]).type = TRIANGLE;
//   (s_objects[7]).matIndex = 1;
//   (s_objects[7]).object = &s_triangles[5];

//   //Left wall red
//   (s_objects[8]).type = TRIANGLE;
//   (s_objects[8]).matIndex = 2;
//   (s_objects[8]).object = &s_triangles[6];

//   (s_objects[9]).type = TRIANGLE;
//   (s_objects[9]).matIndex = 2;
//   (s_objects[9]).object = &s_triangles[7];

//   //Right wall blue
//   (s_objects[10]).type = TRIANGLE;
//   (s_objects[10]).matIndex = 3;
//   (s_objects[10]).object = &s_triangles[8];

//   (s_objects[11]).type = TRIANGLE;
//   (s_objects[11]).matIndex = 3;
//   (s_objects[11]).object = &s_triangles[9];

//   return s_objects;
// }

// //malloc space
// __host__ Material* createMaterials(){
//   Material* s_materials = (Material *) malloc(sizeof(Material) * 6);

//   //Mat 0, diffuse white
//   (s_materials[0]).color.x = 255.0f;
//   (s_materials[0]).color.y = 255.0f;
//   (s_materials[0]).color.z = 255.0f;
//   (s_materials[0]).type = ORIGINAL;
//   (s_materials[0]).metal = 0;
//   (s_materials[0]).specular = 0.0f;
//   (s_materials[0]).lambert = 0.85f;
//   (s_materials[0]).ambient = 0.05f;
//   (s_materials[0]).exponent = 0.0f;

//   //Mat 1, diffuse white
//   (s_materials[1]).color.x = 255.0f;
//   (s_materials[1]).color.y = 255.0f;
//   (s_materials[1]).color.z = 255.0f;
//   (s_materials[1]).type = ORIGINAL;
//   (s_materials[1]).metal = 0;
//   (s_materials[1]).specular = 0.0f;
//   (s_materials[1]).lambert = 0.9f;
//   (s_materials[1]).ambient = 0.05f;
//   (s_materials[1]).exponent = 0.0f;

//   //Mat 2, diffuse red
//   (s_materials[2]).color.x = 255.0f;
//   (s_materials[2]).color.y = 90.0f;
//   (s_materials[2]).color.z = 90.0f;
//   (s_materials[2]).type = ORIGINAL;
//   (s_materials[2]).metal = 0;
//   (s_materials[2]).specular = 0.0f;
//   (s_materials[2]).lambert = 0.9f;
//   (s_materials[2]).ambient = 0.1f;
//   (s_materials[2]).exponent = 0.0f;

//   //Mat 3, diffuse blue
//   (s_materials[3]).color.x = 90.0f;
//   (s_materials[3]).color.y = 90.0f;
//   (s_materials[3]).color.z = 255.0f;
//   (s_materials[3]).type = ORIGINAL;
//   (s_materials[3]).metal = 0;
//   (s_materials[3]).specular = 0.0f;
//   (s_materials[3]).lambert = 0.9f;
//   (s_materials[3]).ambient = 0.1f;
//   (s_materials[3]).exponent = 0.0f;

//   //Mat 4, Mirror
//   (s_materials[4]).color.x = 255.0f;
//   (s_materials[4]).color.y = 255.0f;
//   (s_materials[4]).color.z = 255.0f;
//   (s_materials[4]).type = ORIGINAL;
//   (s_materials[4]).metal = 0;
//   (s_materials[4]).specular = 0.9f;
//   (s_materials[4]).lambert = 0.1f;
//   (s_materials[4]).ambient = 0.0f;
//   (s_materials[4]).exponent = 0.0f;

//   //Mat 5, Gold
//   (s_materials[5]).color.x = 200.0f;
//   (s_materials[5]).color.y = 170.0f;
//   (s_materials[5]).color.z = 60.0f;
//   (s_materials[5]).type = PHONG;
//   (s_materials[5]).metal = 1;
//   (s_materials[5]).specular = 0.5f;
//   (s_materials[5]).lambert = 0.4f;
//   (s_materials[5]).ambient = 0.1f;
//   (s_materials[5]).exponent = 2.0f;

//   return s_materials;
// }


// //malloc space
// __host__ Scene* createScene(Camera* s_camera, Material* s_materials, Object* s_objects, Light* s_lights){
//   Scene* s_scene = (Scene *) malloc(sizeof(Scene) * 1);

//   (s_scene[0]).camera = s_camera;
//   (s_scene[0]).materials = (s_materials);
//   (s_scene[0]).objects = (s_objects);
//   (s_scene[0]).lights = (s_lights);
//   (s_scene[0]).n_lights = 3;
//   (s_scene[0]).n_materials = 6;
//   (s_scene[0]).n_objects = 12;

//   return s_scene;
// }


