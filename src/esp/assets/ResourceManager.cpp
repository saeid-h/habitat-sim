// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "ResourceManager.h"

#include <Corrade/Containers/PointerStl.h>
#include <Corrade/PluginManager/Manager.h>
#include <Corrade/PluginManager/PluginMetadata.h>
#include <Corrade/Utility/Assert.h>
#include <Corrade/Utility/ConfigurationGroup.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Directory.h>
#include <Corrade/Utility/String.h>
#include <Magnum/EigenIntegration/GeometryIntegration.h>
#include <Magnum/EigenIntegration/Integration.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Extensions.h>
#include <Magnum/ImageView.h>
#include <Magnum/Math/FunctionsBatch.h>
#include <Magnum/Math/Range.h>
#include <Magnum/Math/Tags.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshObjectData3D.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/TextureData.h>

#include "esp/geo/geo.h"
#include "esp/gfx/GenericDrawable.h"
#include "esp/io/io.h"
#include "esp/io/json.h"
#include "esp/physics/PhysicsManager.h"
#include "esp/scene/SceneConfiguration.h"
#include "esp/scene/SceneGraph.h"

#include "esp/nav/PathFinder.h"

#ifdef ESP_BUILD_WITH_BULLET
#include "esp/physics/bullet/BulletPhysicsManager.h"
#endif

#include "CollisionMeshData.h"
#include "GenericInstanceMeshData.h"
#include "GenericMeshData.h"
#include "MeshData.h"

#ifdef ESP_BUILD_PTEX_SUPPORT
#include "PTexMeshData.h"
#include "esp/gfx/PTexMeshDrawable.h"
#include "esp/gfx/PTexMeshShader.h"
#endif

namespace Cr = Corrade;
namespace Mn = Magnum;

namespace esp {
namespace assets {
// static constexpr arrays require redundant definitions until C++17
constexpr char ResourceManager::NO_LIGHT_KEY[];
constexpr char ResourceManager::DEFAULT_LIGHTING_KEY[];
constexpr char ResourceManager::DEFAULT_MATERIAL_KEY[];
constexpr char ResourceManager::PER_VERTEX_OBJECT_ID_MATERIAL_KEY[];

ResourceManager::ResourceManager()
    :
#ifdef MAGNUM_BUILD_STATIC
      // avoid using plugins that might depend on different library versions
      importerManager_("nonexistent")
#endif
{
  initDefaultLightSetups();
  initDefaultMaterials();
  buildMapOfPrimTypeConstructors();
}  // namespace assets
void ResourceManager::buildMapOfPrimTypeConstructors() {
  primTypeConstructorMap_["capsule3DSolid"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::CapsulePrimitiveAttributes, false,
          PrimObjTypes::CAPSULE_SOLID>;
  primTypeConstructorMap_["capsule3DWireframe"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::CapsulePrimitiveAttributes, true, PrimObjTypes::CAPSULE_WF>;
  primTypeConstructorMap_["coneSolid"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::ConePrimitiveAttributes, false, PrimObjTypes::CONE_SOLID>;
  primTypeConstructorMap_["coneWireframe"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::ConePrimitiveAttributes, true, PrimObjTypes::CONE_WF>;
  primTypeConstructorMap_["cubeSolid"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::CubePrimitiveAttributes, false, PrimObjTypes::CUBE_SOLID>;
  primTypeConstructorMap_["cubeWireframe"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::CubePrimitiveAttributes, true, PrimObjTypes::CUBE_WF>;
  primTypeConstructorMap_["cylinderSolid"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::CylinderPrimitiveAttributes, false,
          PrimObjTypes::CYLINDER_SOLID>;
  primTypeConstructorMap_["cylinderWireframe"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::CylinderPrimitiveAttributes, true, PrimObjTypes::CYLINDER_WF>;
  primTypeConstructorMap_["icosphereSolid"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::IcospherePrimitiveAttributes, false,
          PrimObjTypes::ICOSPHERE_SOLID>;
  primTypeConstructorMap_["icosphereWireframe"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::IcospherePrimitiveAttributes, true,
          PrimObjTypes::ICOSPHERE_WF>;
  primTypeConstructorMap_["uvSphereSolid"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::UVSpherePrimitiveAttributes, false,
          PrimObjTypes::UVSPHERE_SOLID>;
  primTypeConstructorMap_["uvSphereWireframe"] =
      &ResourceManager::createPrimitiveAttributes<
          assets::UVSpherePrimitiveAttributes, true, PrimObjTypes::UVSPHERE_WF>;
  // no entry added for PrimObjTypes::END_PRIM_OBJ_TYPES

  // build default AbstractPrimitiveAttributes objects
  for (const std::pair<PrimObjTypes, const char*>& elem : PrimitiveNames3DMap) {
    if (elem.first == PrimObjTypes::END_PRIM_OBJ_TYPES) {
      continue;
    }
    auto attr = buildPrimitiveAttributes(elem.second);
    addPrimAssetTemplateToLibrary(attr);
  }
}  // buildMapOfPrimTypeConstructors

void ResourceManager::initDefaultPrimAttributes() {
  // instantiate a primitive importer
  CORRADE_INTERNAL_ASSERT_OUTPUT(
      primImporter_ = importerManager_.loadAndInstantiate("PrimitiveImporter"));
  // necessary for importer to be usable
  primImporter_->openData("");

  // by this point, we should have a GL::Context so load the bb primitive.
  // TODO: replace this completely with standard mesh (i.e. treat the bb
  // wireframe cube no differently than other primivite-based rendered objects)
  auto wfCube = primImporter_->mesh(
      primitiveAssetsTemplateLibrary_["cubeWireframe"]->getPrimObjClassName());
  primitive_meshes_.push_back(
      std::make_unique<Magnum::GL::Mesh>(Magnum::MeshTools::compile(*wfCube)));

  // build default primtive object templates corresponding to given default
  // assets
  for (auto primAsset : primitiveAssetsTemplateLibrary_) {
    buildAndRegisterPrimPhysObjTemplate(primAsset.first);
  }

  LOG(INFO) << "Built primitive asset templates: "
            << std::to_string(primitiveAssetTmpltLibByID_.size());

}  // initDefaultPrimAttributes

bool ResourceManager::loadScene(
    const AssetInfo& info,
    scene::SceneNode* parent, /* = nullptr */
    DrawableGroup* drawables, /* = nullptr */
    const Magnum::ResourceKey& lightSetup /* = Mn::ResourceKey{NO_LIGHT_KEY} */,
    bool splitSemanticMesh /* = true */) {
  // we only compute absolute AABB for every mesh component when loading ptex
  // mesh, or general mesh (e.g., MP3D)
  staticDrawableInfo_.clear();
  if (info.type == AssetType::FRL_PTEX_MESH ||
      info.type == AssetType::MP3D_MESH || info.type == AssetType::UNKNOWN ||
      (info.type == AssetType::INSTANCE_MESH && splitSemanticMesh)) {
    computeAbsoluteAABBs_ = true;
  }

  // scene mesh loading
  bool meshSuccess = true;
  if (info.filepath.compare(EMPTY_SCENE) != 0) {
    if (!io::exists(info.filepath)) {
      LOG(ERROR) << "Cannot load from file " << info.filepath;
      meshSuccess = false;
    } else {
      if (info.type == AssetType::INSTANCE_MESH) {
        meshSuccess =
            loadInstanceMeshData(info, parent, drawables, splitSemanticMesh);
      } else if (info.type == AssetType::FRL_PTEX_MESH) {
        meshSuccess = loadPTexMeshData(info, parent, drawables);
      } else if (info.type == AssetType::SUNCG_SCENE) {
        meshSuccess = loadSUNCGHouseFile(info, parent, drawables);
      } else if (info.type == AssetType::MP3D_MESH) {
        meshSuccess = loadGeneralMeshData(info, parent, drawables, lightSetup);
      } else {
        // Unknown type, just load general mesh data
        meshSuccess = loadGeneralMeshData(info, parent, drawables, lightSetup);
      }
      // add a scene attributes for this filename or modify the existing one
      if (meshSuccess) {
        const bool physSceneExists =
            physicsSceneLibrary_.count(info.filepath) > 0;
        if (!physSceneExists) {
          auto physScenLib = PhysicsSceneAttributes::create(info.filepath);
          physicsSceneLibrary_.emplace(info.filepath, physScenLib);
        }
        physicsSceneLibrary_.at(info.filepath)
            ->setRenderAssetHandle(info.filepath);
      }
    }
  } else {
    LOG(INFO) << "Loading empty scene";
    // EMPTY_SCENE (ie. "NONE") string indicates desire for an empty scene (no
    // scene mesh): welcome to the void
  }

  // compute the absolute transformation for each static drawables
  if (meshSuccess && parent && computeAbsoluteAABBs_) {
    if (info.type == AssetType::FRL_PTEX_MESH) {
#ifdef ESP_BUILD_PTEX_SUPPORT
      // retrieve the ptex mesh data
      const std::string& filename = info.filepath;
      CORRADE_ASSERT(resourceDict_.count(filename) != 0,
                     "ResourceManager::loadScene: ptex mesh is not loaded.",
                     false);
      const MeshMetaData& metaData = getMeshMetaData(filename);
      CORRADE_ASSERT(
          metaData.meshIndex.first == metaData.meshIndex.second,
          "ResourceManager::loadScene: ptex mesh is not loaded correctly.",
          false);

      computePTexMeshAbsoluteAABBs(*meshes_[metaData.meshIndex.first]);
#endif
    } else if (info.type == AssetType::MP3D_MESH ||
               info.type == AssetType::UNKNOWN) {
      computeGeneralMeshAbsoluteAABBs();
    } else if (info.type == AssetType::INSTANCE_MESH) {
      computeInstanceMeshAbsoluteAABBs();
    }
  }

  if (computeAbsoluteAABBs_) {
    computeAbsoluteAABBs_ = false;
    // this is to prevent it from being misused in the future
    staticDrawableInfo_.clear();
  }

  return meshSuccess;
}  // ResourceManager::loadScene

void ResourceManager::loadObjectTemplates(
    const std::vector<std::string>& tmpltFilenames) {
  for (auto objPhysPropertiesFilename : tmpltFilenames) {
    LOG(INFO) << "loading object: " << objPhysPropertiesFilename;
    parseAndLoadPhysObjTemplate(objPhysPropertiesFilename);
  }
  LOG(INFO) << "loaded object templates: "
            << std::to_string(physicsFileObjTmpltLibByID_.size());
}  // ResourceManager::loadObjectTemplates

void ResourceManager::initPhysicsManager(
    std::shared_ptr<physics::PhysicsManager>& physicsManager,
    PhysicsManagerAttributes::ptr physicsManagerAttributes) {
  //! PHYSICS INIT: Use the above config to initialize physics engine
  bool defaultToNoneSimulator = true;
  if (physicsManagerAttributes->getSimulator().compare("bullet") == 0) {
#ifdef ESP_BUILD_WITH_BULLET
    physicsManager.reset(
        new physics::BulletPhysicsManager(*this, physicsManagerAttributes));
    defaultToNoneSimulator = false;
#else
    LOG(WARNING)
        << ":\n---\nPhysics was enabled and Bullet physics engine was "
           "specified, but the project is built without Bullet support. "
           "Objects added to the scene will be restricted to kinematic updates "
           "only. Reinstall with --bullet to enable Bullet dynamics.\n---";
#endif
  }
  // reset to base PhysicsManager to override previous as default behavior
  // if the desired simulator is not supported reset to "none" in metaData
  if (defaultToNoneSimulator) {
    physicsManager.reset(
        new physics::PhysicsManager(*this, physicsManagerAttributes));
    physicsManagerAttributes->setSimulator("none");
  }
  // build default primitive asset templates, and default primitive object
  // templates
  initDefaultPrimAttributes();
  // load object templates from sceneMetaData list...
  loadObjectTemplates(
      physicsManagerAttributes->getStringGroup("objectLibraryPaths"));
}  // ResourceManager::initPhysicsManager

// this will instance a physics manager based on the passed physics config file
// name
std::shared_ptr<physics::PhysicsManager> ResourceManager::buildPhysicsManager(
    std::string physicsFilename /* data/default.phys_scene_config.json */) {
  // In-memory representation of scene meta data
  PhysicsManagerAttributes::ptr physicsManagerAttributes =
      loadPhysicsConfig(physicsFilename);
  physicsManagerLibrary_.emplace(physicsFilename, physicsManagerAttributes);
  std::shared_ptr<physics::PhysicsManager> _physicsManager = nullptr;
  // initialize physics manager with derived physics manager attributes
  initPhysicsManager(_physicsManager, physicsManagerAttributes);

  return _physicsManager;
}  // ResourceManager::buildPhysicsManager

//! (1) Read config and set physics timestep
//! (2) loadScene() with PhysicsSceneMetaData
// TODO (JH): this function seems to entangle certain physicsManager functions
bool ResourceManager::loadScene(
    const AssetInfo& info,
    std::shared_ptr<physics::PhysicsManager>& _physicsManager,
    scene::SceneNode* parent,              /* = nullptr */
    DrawableGroup* drawables,              /* = nullptr */
    const Magnum::ResourceKey& lightSetup, /* = Mn::ResourceKey{NO_LIGHT_KEY} */
    std::string physicsFilename /* data/default.phys_scene_config.json */) {
  // In-memory representation of scene meta data
  PhysicsManagerAttributes::ptr physicsManagerAttributes =
      loadPhysicsConfig(physicsFilename);
  physicsManagerLibrary_.emplace(physicsFilename, physicsManagerAttributes);
  return loadPhysicsScene(info, _physicsManager, physicsManagerAttributes,
                          parent, drawables, lightSetup);
}

// TODO: kill existing scene mesh drawables, nodes, etc... (all but meshes in
// memory?)
//! (1) load scene mesh
//! (2) add drawable (if parent and drawables != nullptr)
//! (3) consume PhysicsSceneMetaData to initialize physics simulator
//! (4) create scene collision mesh if possible
bool ResourceManager::loadPhysicsScene(
    const AssetInfo& info,
    std::shared_ptr<physics::PhysicsManager>& _physicsManager,
    PhysicsManagerAttributes::ptr physicsManagerAttributes,
    scene::SceneNode* parent, /* = nullptr */
    DrawableGroup* drawables, /* = nullptr */
    const Magnum::ResourceKey&
        lightSetup /* = Mn::ResourceKey{NO_LIGHT_KEY} */) {
  // default scene mesh loading
  bool meshSuccess = loadScene(info, parent, drawables, lightSetup);
  // (re)init physics manager
  initPhysicsManager(_physicsManager, physicsManagerAttributes);

  // initialize the physics simulator
  _physicsManager->initPhysics(parent);

  if (!meshSuccess) {
    LOG(ERROR) << "Physics manager loaded. Scene mesh load failed, aborting "
                  "scene initialization.";
    return meshSuccess;
  }

  const bool physSceneExists = physicsSceneLibrary_.count(info.filepath) > 0;
  if (!physSceneExists) {
    auto physScenLib = PhysicsSceneAttributes::create(info.filepath);
    physicsSceneLibrary_.emplace(info.filepath, physScenLib);
  }
  // TODO: enable loading of multiple scenes from file and storing individual
  // parameters instead of scene properties in manager global config

  physicsSceneLibrary_.at(info.filepath)
      ->setFrictionCoefficient(
          physicsManagerAttributes->getFrictionCoefficient());
  physicsSceneLibrary_.at(info.filepath)
      ->setRestitutionCoefficient(
          physicsManagerAttributes->getRestitutionCoefficient());

  physicsSceneLibrary_.at(info.filepath)->setRenderAssetHandle(info.filepath);
  physicsSceneLibrary_.at(info.filepath)
      ->setCollisionAssetHandle(info.filepath);

  //! CONSTRUCT SCENE
  const std::string& filename = info.filepath;
  // if we have a scene mesh, add it as a collision object
  if (filename.compare(EMPTY_SCENE) != 0) {
    const MeshMetaData& metaData = getMeshMetaData(filename);
    auto indexPair = metaData.meshIndex;
    int start = indexPair.first;
    int end = indexPair.second;

    //! Collect collision mesh group
    std::vector<CollisionMeshData> meshGroup;
    for (int mesh_i = start; mesh_i <= end; ++mesh_i) {
      // PLY Instance mesh
      if (info.type == AssetType::INSTANCE_MESH) {
        GenericInstanceMeshData* insMeshData =
            dynamic_cast<GenericInstanceMeshData*>(meshes_[mesh_i].get());
        CollisionMeshData& meshData = insMeshData->getCollisionMeshData();
        meshGroup.push_back(meshData);
      }

      // GLB Mesh
      else if (info.type == AssetType::MP3D_MESH ||
               info.type == AssetType::UNKNOWN) {
        GenericMeshData* gltfMeshData =
            dynamic_cast<GenericMeshData*>(meshes_[mesh_i].get());
        if (gltfMeshData == nullptr) {
          Corrade::Utility::Debug()
              << "AssetInfo::AssetType type error: unsupported physical type, "
                 "aborting. Try running without \"--enable-physics\" and "
                 "consider logging an issue.";
          return false;
        }
        CollisionMeshData& meshData = gltfMeshData->getCollisionMeshData();
        meshGroup.push_back(meshData);
      }
    }

    // add meshgroup to collision mesh groups
    collisionMeshGroups_.emplace(info.filepath, meshGroup);
    //! Initialize collision mesh
    bool sceneSuccess = _physicsManager->addScene(
        physicsSceneLibrary_.at(info.filepath), meshGroup);
    if (!sceneSuccess) {
      return false;
    }
  }

  return meshSuccess;
}

std::vector<std::string> ResourceManager::buildObjectConfigPaths(
    const std::string& path) {
  std::vector<std::string> paths;

  namespace Directory = Cr::Utility::Directory;
  std::string objPhysPropertiesFilename = path;
  if (!Corrade::Utility::String::endsWith(objPhysPropertiesFilename,
                                          ".phys_properties.json")) {
    objPhysPropertiesFilename = path + ".phys_properties.json";
  }
  const bool dirExists = Directory::isDirectory(path);
  const bool fileExists = Directory::exists(objPhysPropertiesFilename);

  if (!dirExists && !fileExists) {
    LOG(WARNING) << "Cannot find " << path << " or "
                 << objPhysPropertiesFilename << ". Aborting parse.";
    return paths;
  }

  if (fileExists) {
    paths.push_back(objPhysPropertiesFilename);
  }

  if (dirExists) {
    LOG(INFO) << "Parsing object library directory: " + path;
    for (auto& file : Directory::list(path, Directory::Flag::SortAscending)) {
      std::string absoluteSubfilePath = Directory::join(path, file);
      if (Cr::Utility::String::endsWith(absoluteSubfilePath,
                                        ".phys_properties.json")) {
        paths.push_back(absoluteSubfilePath);
      }
    }
  }

  return paths;
}  // buildObjectConfigPaths

PhysicsManagerAttributes::ptr ResourceManager::loadPhysicsConfig(
    std::string physicsFilename) {
  CHECK(Cr::Utility::Directory::exists(physicsFilename));

  // Load the global scene config JSON here
  io::JsonDocument scenePhysicsConfig = io::parseJsonFile(physicsFilename);
  // In-memory representation of scene meta data
  PhysicsManagerAttributes::ptr physicsManagerAttributes =
      PhysicsManagerAttributes::create(physicsFilename);

  // load the simulator preference
  // default is "none" simulator
  if (scenePhysicsConfig.HasMember("physics simulator")) {
    if (scenePhysicsConfig["physics simulator"].IsString()) {
      physicsManagerAttributes->setSimulator(
          scenePhysicsConfig["physics simulator"].GetString());
    }
  }

  // load the physics timestep
  if (scenePhysicsConfig.HasMember("timestep")) {
    if (scenePhysicsConfig["timestep"].IsNumber()) {
      physicsManagerAttributes->setTimestep(
          scenePhysicsConfig["timestep"].GetDouble());
    }
  }

  if (scenePhysicsConfig.HasMember("friction coefficient") &&
      scenePhysicsConfig["friction coefficient"].IsNumber()) {
    physicsManagerAttributes->setFrictionCoefficient(
        scenePhysicsConfig["friction coefficient"].GetDouble());
  } else {
    LOG(ERROR) << " Invalid value in scene config - friction coefficient";
  }

  if (scenePhysicsConfig.HasMember("restitution coefficient") &&
      scenePhysicsConfig["restitution coefficient"].IsNumber()) {
    physicsManagerAttributes->setRestitutionCoefficient(
        scenePhysicsConfig["restitution coefficient"].GetDouble());
  } else {
    LOG(ERROR) << " Invalid value in scene config - restitution coefficient";
  }

  // load gravity
  if (scenePhysicsConfig.HasMember("gravity")) {
    if (scenePhysicsConfig["gravity"].IsArray()) {
      Magnum::Vector3 grav;
      for (rapidjson::SizeType i = 0; i < scenePhysicsConfig["gravity"].Size();
           i++) {
        if (!scenePhysicsConfig["gravity"][i].IsNumber()) {
          // invalid config
          LOG(ERROR) << "Invalid value in physics gravity array";
          break;
        } else {
          grav[i] = scenePhysicsConfig["gravity"][i].GetDouble();
        }
      }
      physicsManagerAttributes->setGravity(grav);
    }
  }

  // load the rigid object library metadata (no physics init yet...)
  if (!scenePhysicsConfig.HasMember("rigid object paths") ||
      !scenePhysicsConfig["rigid object paths"].IsArray()) {
    return physicsManagerAttributes;
  }

  std::string configDirectory =
      physicsFilename.substr(0, physicsFilename.find_last_of("/"));

  const auto& paths = scenePhysicsConfig["rigid object paths"];
  for (rapidjson::SizeType i = 0; i < paths.Size(); i++) {
    if (!paths[i].IsString()) {
      LOG(ERROR) << "Invalid value in physics scene config -rigid object "
                    "library- array "
                 << i;
      continue;
    }

    std::string absolutePath =
        Cr::Utility::Directory::join(configDirectory, paths[i].GetString());
    std::vector<std::string> validConfigPaths =
        buildObjectConfigPaths(absolutePath);
    for (auto& path : validConfigPaths) {
      physicsManagerAttributes->addStringToGroup("objectLibraryPaths", path);
    }
  }

  return physicsManagerAttributes;
}  // loadPhysicsConfig

bool ResourceManager::loadObjectMeshDataFromFile(
    const std::string& filename,
    const std::string& objectTemplateHandle,
    const std::string& meshType,
    const bool requiresLighting) {
  bool success = false;
  if (!filename.empty()) {
    AssetInfo meshInfo{AssetType::UNKNOWN, filename};
    meshInfo.requiresLighting = requiresLighting;
    success = loadGeneralMeshData(meshInfo);
    if (!success) {
      LOG(ERROR) << "Failed to load a physical object's " << meshType
                 << " mesh: " << objectTemplateHandle << ", " << filename;
    }
  }
  return success;
}  // loadObjectMeshDataFromFile

int ResourceManager::addObjTemplateToLibrary(
    PhysicsObjectAttributes::ptr objectTemplate,
    const std::string& objectTemplateHandle,
    std::map<int, std::string>& mapOfNames) {
  // add object template ID to physicsObjTemplateLibrary_
  bool isNotPresent =
      (physicsObjTemplateLibrary_.count(objectTemplateHandle) == 0);
  int objectTemplateID;
  if (isNotPresent) {
    objectTemplateID = physicsObjTemplateLibrary_.size();
    objectTemplate->setObjectTemplateID(objectTemplateID);
  } else {
    objectTemplateID = objectTemplate->getObjectTemplateID();
  }
  // if is present, will replace present template with new template - templates
  // are expected to be 1-to-1 with handles (each unique handle always describes
  // the same unique template)
  physicsObjTemplateLibrary_.emplace(objectTemplateHandle, objectTemplate);
  physicsTemplatesLibByID_.emplace(objectTemplateID, objectTemplateHandle);
  // save reference of specific type of template being built
  mapOfNames.emplace(objectTemplateID, objectTemplateHandle);
  return objectTemplateID;
}  // addObjTemplateToLibrary

int ResourceManager::addPrimAssetTemplateToLibrary(
    AbstractPrimitiveAttributes::ptr primTemplate) {
  // add prim asset template ID to physicsObjTemplateLibrary_
  auto primHandle = primTemplate->getOriginHandle();

  bool isNotPresent = (primitiveAssetsTemplateLibrary_.count(primHandle) == 0);
  int primAssetTemplateID;
  if (isNotPresent) {
    primAssetTemplateID = primitiveAssetsTemplateLibrary_.size();
    primTemplate->setAssetTemplateID(primAssetTemplateID);
  } else {
    primAssetTemplateID = primTemplate->getAssetTemplateID();
  }
  // if is present, will replace present template with new template - templates
  // are expected to be 1-to-1 with handles (each unique handle always describes
  // the same unique template)
  primitiveAssetsTemplateLibrary_.emplace(primHandle, primTemplate);
  primitiveAssetTmpltLibByID_.emplace(primAssetTemplateID, primHandle);

  return primAssetTemplateID;
}  // addPrimAssetTemplateToLibrary

int ResourceManager::loadObjectTemplate(
    PhysicsObjectAttributes::ptr objectTemplate,
    const std::string& objectTemplateHandle) {
  if (physicsObjTemplateLibrary_.count(objectTemplateHandle) != 0) {
    return physicsObjTemplateLibrary_.at(objectTemplateHandle)
        ->getObjectTemplateID();
  }
  CHECK(objectTemplate->hasValue("renderAssetHandle"));
  // In case not constructed with handle as parameter
  objectTemplate->setOriginHandle(objectTemplateHandle);

  // load/check_for render and collision mesh metadata
  bool requiresLighting = objectTemplate->getRequiresLighting();

  //! Get render and collision mesh names
  // AssetInfo renderMeshinfo;
  std::string renderMeshFilename = objectTemplate->getRenderAssetHandle();
  bool renderMeshSuccess = loadObjectMeshDataFromFile(
      renderMeshFilename, objectTemplateHandle, "render", requiresLighting);

  // if render mesh failed, might have to generate lighting data for collision
  // mesh since we will use it to render
  // AssetInfo collisionMeshinfo;
  std::string collisionMeshFilename = objectTemplate->getCollisionAssetHandle();
  bool collisionMeshSuccess = loadObjectMeshDataFromFile(
      collisionMeshFilename, objectTemplateHandle, "collision",
      !renderMeshSuccess && requiresLighting);

  if (!renderMeshSuccess && !collisionMeshSuccess) {
    // we only allow objects with SOME mesh file. Failing
    // both loads or having no mesh will cancel the load.
    LOG(ERROR) << "Failed to load a physical object: no meshes...: "
               << objectTemplateHandle;
    return ID_UNDEFINED;
  }

  // handle one missing mesh
  if (!renderMeshSuccess) {
    objectTemplate->setRenderAssetHandle(collisionMeshFilename);
  }
  if (!collisionMeshSuccess) {
    objectTemplate->setCollisionAssetHandle(renderMeshFilename);
  }

  const auto collisionAssetHandle = objectTemplate->getCollisionAssetHandle();

  // set collision mesh data
  const MeshMetaData& meshMetaData = getMeshMetaData(collisionAssetHandle);

  int start = meshMetaData.meshIndex.first;
  int end = meshMetaData.meshIndex.second;
  //! Gather mesh components for meshGroup data
  std::vector<CollisionMeshData> meshGroup;
  for (int mesh_i = start; mesh_i <= end; ++mesh_i) {
    GenericMeshData* gltfMeshData =
        dynamic_cast<GenericMeshData*>(meshes_[mesh_i].get());
    CollisionMeshData& meshData = gltfMeshData->getCollisionMeshData();
    meshGroup.push_back(meshData);
  }
  collisionMeshGroups_.emplace(collisionAssetHandle, meshGroup);

  // add object template to template library
  int objectTemplateID = addObjTemplateToLibrary(
      objectTemplate, objectTemplateHandle, physicsFileObjTmpltLibByID_);

  return objectTemplateID;
}  // loadObjectTemplate

std::string ResourceManager::getRandomTemplateHandlePerType(
    const std::map<int, std::string>& mapOfHandles,
    const std::string& type) const {
  int numVals = mapOfHandles.size();
  if (numVals == 0) {
    LOG(ERROR) << "Attempting to get a random " << type
               << "object template handle but none are loaded;Aboring";
    return "";
  }
  int randIDX = rand() % numVals;

  std::string res;
  for (std::pair<std::map<int, std::string>::const_iterator, int> iter(
           mapOfHandles.begin(), 0);
       (iter.first != mapOfHandles.end() && iter.second <= randIDX);
       ++iter.first, ++iter.second) {
    res = iter.first->second;
  }
  return res;
}  // getRandomTemplateHandlePerType

std::vector<std::string> ResourceManager::getTemplateHandlesBySubStringPerType(
    const std::map<int, std::string>& mapOfHandles,
    const std::string& subStr) const {
  std::vector<std::string> res;
  // if empty return empty vector
  if (mapOfHandles.size() == 0) {
    return res;
  }
  // if search string is empty, return all values
  if (subStr.length() == 0) {
    for (auto elem : mapOfHandles) {
      res.push_back(elem.second);
    }
    return res;
  }
  // build search criteria
  std::string strToLookFor(subStr);

  int strSize = strToLookFor.length();
  // force lowercase
  std::transform(strToLookFor.begin(), strToLookFor.end(), strToLookFor.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  for (std::map<int, std::string>::const_iterator iter = mapOfHandles.begin();
       iter != mapOfHandles.end(); ++iter) {
    std::string key(iter->second);
    // be sure that key is big enough to search in (otherwise find has undefined
    // behavior)
    if (key.length() < strSize) {
      continue;
    }
    // force lowercase
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (std::string::npos != key.find(strToLookFor)) {
      res.push_back(iter->second);
    }
  }
  return res;
}  // getTemplateHandlesBySubString

// load object template from config filename
int ResourceManager::parseAndLoadPhysObjTemplate(
    const std::string& objPhysConfigFilename) {
  // check for duplicate load
  const bool objTemplateExists =
      physicsObjTemplateLibrary_.count(objPhysConfigFilename) > 0;
  if (objTemplateExists) {
    return physicsObjTemplateLibrary_.at(objPhysConfigFilename)
        ->getObjectTemplateID();
  }

  // 1. parse the config file
  io::JsonDocument objPhysicsConfig;
  if (io::exists(objPhysConfigFilename)) {
    try {
      objPhysicsConfig = io::parseJsonFile(objPhysConfigFilename);
    } catch (...) {
      LOG(ERROR) << "Failed to parse JSON: " << objPhysConfigFilename
                 << ". Aborting loadObject.";
      return ID_UNDEFINED;
    }
  } else {
    LOG(ERROR) << "File " << objPhysConfigFilename
               << " does not exist. Aborting loadObject.";
    return ID_UNDEFINED;
  }

  // 2. construct a physicsObjectMetaData
  auto physicsObjectAttributes =
      PhysicsObjectAttributes::create(objPhysConfigFilename);

  // NOTE: these paths should be relative to the properties file
  std::string propertiesFileDirectory =
      objPhysConfigFilename.substr(0, objPhysConfigFilename.find_last_of("/"));

  // 3. load physical properties to override defaults (set in
  // PhysicsObjectMetaData.h) load the mass
  if (objPhysicsConfig.HasMember("mass")) {
    if (objPhysicsConfig["mass"].IsNumber()) {
      physicsObjectAttributes->setMass(objPhysicsConfig["mass"].GetDouble());
    }
  }

  // optional set bounding box as collision object
  if (objPhysicsConfig.HasMember("use bounding box for collision")) {
    if (objPhysicsConfig["use bounding box for collision"].IsBool()) {
      physicsObjectAttributes->setBoundingBoxCollisions(
          objPhysicsConfig["use bounding box for collision"].GetBool());
    }
  }

  // load the center of mass (in the local frame of the object)
  // if COM is provided, use it for mesh shift
  if (objPhysicsConfig.HasMember("COM")) {
    if (objPhysicsConfig["COM"].IsArray()) {
      Magnum::Vector3 COM;
      for (rapidjson::SizeType i = 0; i < objPhysicsConfig["COM"].Size(); i++) {
        if (!objPhysicsConfig["COM"][i].IsNumber()) {
          // invalid config
          LOG(ERROR) << " Invalid value in object physics config - COM array";
          break;
        } else {
          COM[i] = objPhysicsConfig["COM"][i].GetDouble();
        }
      }
      physicsObjectAttributes->setCOM(COM);
      // set a flag which we can find later so we don't override the desired
      // COM with BB center.
      physicsObjectAttributes->setBool("COM_provided", true);
    }
  }

  // scaling
  if (objPhysicsConfig.HasMember("scale")) {
    if (objPhysicsConfig["scale"].IsArray()) {
      Magnum::Vector3 scale;
      for (rapidjson::SizeType i = 0; i < objPhysicsConfig["scale"].Size();
           i++) {
        if (!objPhysicsConfig["scale"][i].IsNumber()) {
          // invalid config
          LOG(ERROR) << " Invalid value in object physics config - scale array";
          break;
        } else {
          scale[i] = objPhysicsConfig["scale"][i].GetDouble();
        }
      }
      physicsObjectAttributes->setScale(scale);
    }
  }

  // load the inertia diagonal
  if (objPhysicsConfig.HasMember("inertia")) {
    if (objPhysicsConfig["inertia"].IsArray()) {
      Magnum::Vector3 inertia;
      for (rapidjson::SizeType i = 0; i < objPhysicsConfig["inertia"].Size();
           i++) {
        if (!objPhysicsConfig["inertia"][i].IsNumber()) {
          // invalid config
          LOG(ERROR)
              << " Invalid value in object physics config - inertia array";
          break;
        } else {
          inertia[i] = objPhysicsConfig["inertia"][i].GetDouble();
        }
      }
      physicsObjectAttributes->setInertia(inertia);
    }
  }

  // load the friction coefficient
  if (objPhysicsConfig.HasMember("friction coefficient")) {
    if (objPhysicsConfig["friction coefficient"].IsNumber()) {
      physicsObjectAttributes->setFrictionCoefficient(
          objPhysicsConfig["friction coefficient"].GetDouble());
    } else {
      LOG(ERROR)
          << " Invalid value in object physics config - friction coefficient";
    }
  }

  // load the restitution coefficient
  if (objPhysicsConfig.HasMember("restitution coefficient")) {
    if (objPhysicsConfig["restitution coefficient"].IsNumber()) {
      physicsObjectAttributes->setRestitutionCoefficient(
          objPhysicsConfig["restitution coefficient"].GetDouble());
    } else {
      LOG(ERROR) << " Invalid value in object physics config - restitution "
                    "coefficient";
    }
  }

  //! Get collision configuration options if specified
  if (objPhysicsConfig.HasMember("join collision meshes")) {
    if (objPhysicsConfig["join collision meshes"].IsBool()) {
      physicsObjectAttributes->setJoinCollisionMeshes(
          objPhysicsConfig["join collision meshes"].GetBool());
    } else {
      LOG(ERROR) << " Invalid value in object physics config - join "
                    "collision meshes";
    }
  }

  // if object will be flat or phong shaded
  if (objPhysicsConfig.HasMember("requires lighting")) {
    if (objPhysicsConfig["requires lighting"].IsBool()) {
      physicsObjectAttributes->setRequiresLighting(
          objPhysicsConfig["requires lighting"].GetBool());
    } else {
      LOG(ERROR)
          << " Invalid value in object physics config - requires lighting";
    }
  }

  // 4. parse render and collision mesh filepaths
  std::string renderMeshFilename = "";
  std::string collisionMeshFilename = "";

  if (objPhysicsConfig.HasMember("render mesh")) {
    if (objPhysicsConfig["render mesh"].IsString()) {
      renderMeshFilename = Cr::Utility::Directory::join(
          propertiesFileDirectory, objPhysicsConfig["render mesh"].GetString());
    } else {
      LOG(ERROR) << " Invalid value in object physics config - render mesh";
    }
  }
  if (objPhysicsConfig.HasMember("collision mesh")) {
    if (objPhysicsConfig["collision mesh"].IsString()) {
      collisionMeshFilename = Cr::Utility::Directory::join(
          propertiesFileDirectory,
          objPhysicsConfig["collision mesh"].GetString());
    } else {
      LOG(ERROR) << " Invalid value in object physics config - collision mesh";
    }
  }

  physicsObjectAttributes->setRenderAssetHandle(renderMeshFilename);
  physicsObjectAttributes->setCollisionAssetHandle(collisionMeshFilename);

  // 5. load the parsed file into the library
  return loadObjectTemplate(physicsObjectAttributes, objPhysConfigFilename);
}  // parseAndLoadPhysObjTemplate

int ResourceManager::buildAndRegisterPrimPhysObjTemplate(
    const std::string& primAssetHandle) {
  // verify that a primitive asset with the given handle exists
  if (primitiveAssetsTemplateLibrary_.count(primAssetHandle) == 0) {
    LOG(WARNING) << " No primitive with handle '" << primAssetHandle
                 << "' exists so cannot build physical object.  Aborting.";
    return ID_UNDEFINED;
  }

  // verify that a template with this asset does not exist
  if (physicsObjTemplateLibrary_.count(primAssetHandle) > 0) {
    return physicsObjTemplateLibrary_.at(primAssetHandle)
        ->getObjectTemplateID();
  }

  // construct a physicsObjectMetaData
  auto physicsObjectAttributes =
      PhysicsObjectAttributes::create(primAssetHandle);
  // set default for primitives to not use mesh collisions
  physicsObjectAttributes->setUseMeshCollision(false);
  // set render mesh handle
  physicsObjectAttributes->setRenderAssetHandle(primAssetHandle);
  // set collision mesh/primitive handle
  physicsObjectAttributes->setCollisionAssetHandle(primAssetHandle);
  // NOTE to eventually use mesh collisions with primitive objects, a collision
  // primitive mesh needs to be configured and set in MeshMetaData and
  // CollisionMesh

  // set margin to be 0

  physicsObjectAttributes->setMargin(0.0);
  // make smaller
  const Magnum::Vector3 scale(0.1, 0.1, 0.1);
  physicsObjectAttributes->setScale(scale);

  // add object template to all appropriate libraries
  int objectTemplateID = addObjTemplateToLibrary(
      physicsObjectAttributes, primAssetHandle, physicsSynthObjTmpltLibByID_);

  return objectTemplateID;
}  // ResourceManager::buildAndRegisterPrimPhysObjTemplate

std::string ResourceManager::getObjectTemplateHandle(
    const int objectTemplateID) const {
  const bool physTemplateExists =
      physicsTemplatesLibByID_.count(objectTemplateID) > 0;
  if (!physTemplateExists) {
    Corrade::Utility::Debug()
        << "ResourceManager::getObjectTemplateHandle - Aborting. "
           "No loaded or primitive template with index "
        << objectTemplateID << " exists.";
    return "";
  }
  return physicsTemplatesLibByID_.at(objectTemplateID);
}  // getObjectTemplateHandle

Magnum::Range3D ResourceManager::computeMeshBB(BaseMesh* meshDataGL) {
  CollisionMeshData& meshData = meshDataGL->getCollisionMeshData();
  return Magnum::Range3D{
      Magnum::Math::minmax<Magnum::Vector3>(meshData.positions)};
}

#ifdef ESP_BUILD_PTEX_SUPPORT
void ResourceManager::computePTexMeshAbsoluteAABBs(BaseMesh& baseMesh) {
  std::vector<Mn::Matrix4> absTransforms = computeAbsoluteTransformations();

  CORRADE_ASSERT(absTransforms.size() == staticDrawableInfo_.size(),
                 "ResourceManager::computePTexMeshAbsoluteAABBs: number of "
                 "transformations does not match number of drawables.", );

  // obtain the sub-meshes within the ptex mesh
  PTexMeshData& ptexMeshData = dynamic_cast<PTexMeshData&>(baseMesh);
  const std::vector<PTexMeshData::MeshData>& submeshes = ptexMeshData.meshes();

  for (uint32_t iEntry = 0; iEntry < absTransforms.size(); ++iEntry) {
    // convert std::vector<vec3f> to std::vector<Mn::Vector3>
    const PTexMeshData::MeshData& submesh =
        submeshes[staticDrawableInfo_[iEntry].meshID];
    std::vector<Mn::Vector3> pos{submesh.vbo.begin(), submesh.vbo.end()};

    // transform the vertex positions to the world space
    Mn::MeshTools::transformPointsInPlace(absTransforms[iEntry], pos);

    scene::SceneNode& node = staticDrawableInfo_[iEntry].node;
    node.setAbsoluteAABB(Mn::Range3D{Mn::Math::minmax<Mn::Vector3>(pos)});
  }
}
#endif

void ResourceManager::computeGeneralMeshAbsoluteAABBs() {
  std::vector<Mn::Matrix4> absTransforms = computeAbsoluteTransformations();

  CORRADE_ASSERT(absTransforms.size() == staticDrawableInfo_.size(),
                 "ResourceManager::computeGeneralMeshAbsoluteAABBs: number of "
                 "transforms does not match number of drawables.", );

  for (uint32_t iEntry = 0; iEntry < absTransforms.size(); ++iEntry) {
    const uint32_t meshID = staticDrawableInfo_[iEntry].meshID;

    Corrade::Containers::Optional<Magnum::Trade::MeshData>& meshData =
        meshes_[meshID]->getMeshData();
    CORRADE_ASSERT(meshData,
                   "ResourceManager::computeGeneralMeshAbsoluteAABBs: the "
                   "empty mesh data", );

    // a vector to store the min, max pos for the aabb of every position array
    std::vector<Mn::Vector3> bbPos;

    // transform the vertex positions to the world space, compute the aabb for
    // each position array
    for (uint32_t jArray = 0;
         jArray < meshData->attributeCount(Mn::Trade::MeshAttribute::Position);
         ++jArray) {
      Cr::Containers::Array<Mn::Vector3> pos =
          meshData->positions3DAsArray(jArray);
      Mn::MeshTools::transformPointsInPlace(absTransforms[iEntry], pos);

      std::pair<Mn::Vector3, Mn::Vector3> bb = Mn::Math::minmax(pos);
      bbPos.push_back(std::move(bb.first));
      bbPos.push_back(std::move(bb.second));
    }

    // locate the scene node which contains the current drawable
    scene::SceneNode& node = staticDrawableInfo_[iEntry].node;

    // set the absolute axis aligned bounding box
    node.setAbsoluteAABB(Mn::Range3D{Mn::Math::minmax<Mn::Vector3>(bbPos)});

  }  // iEntry
}

void ResourceManager::computeInstanceMeshAbsoluteAABBs() {
  std::vector<Mn::Matrix4> absTransforms = computeAbsoluteTransformations();

  CORRADE_ASSERT(
      absTransforms.size() == staticDrawableInfo_.size(),
      "ResourceManager::computeInstancelMeshAbsoluteAABBs: number of "
      "transforms does not match number of drawables.", );

  for (size_t iEntry = 0; iEntry < absTransforms.size(); ++iEntry) {
    const uint32_t meshID = staticDrawableInfo_[iEntry].meshID;

    // convert std::vector<vec3f> to std::vector<Mn::Vector3>
    const std::vector<vec3f>& vertexPositions =
        dynamic_cast<GenericInstanceMeshData&>(*meshes_[meshID])
            .getVertexBufferObjectCPU();
    std::vector<Mn::Vector3> transformedPositions{vertexPositions.begin(),
                                                  vertexPositions.end()};

    Mn::MeshTools::transformPointsInPlace(absTransforms[iEntry],
                                          transformedPositions);

    scene::SceneNode& node = staticDrawableInfo_[iEntry].node;
    node.setAbsoluteAABB(
        Mn::Range3D{Mn::Math::minmax<Mn::Vector3>(transformedPositions)});
  }  // iEntry
}

std::vector<Mn::Matrix4> ResourceManager::computeAbsoluteTransformations() {
  // sanity check
  if (staticDrawableInfo_.size() == 0) {
    return {};
  }

  // basic assumption is that all the drawables are in the same scene;
  // so use the 1st element in the vector to obtain this scene
  auto* scene = dynamic_cast<MagnumScene*>(staticDrawableInfo_[0].node.scene());

  CORRADE_ASSERT(scene != nullptr,
                 "ResourceManager::computeAbsoluteTransformations: the node is "
                 "not attached to any scene graph.",
                 {});

  // collect all drawable objects
  std::vector<std::reference_wrapper<MagnumObject>> objects;
  objects.reserve(staticDrawableInfo_.size());
  std::transform(staticDrawableInfo_.begin(), staticDrawableInfo_.end(),
                 std::back_inserter(objects),
                 [](const StaticDrawableInfo& info) -> MagnumObject& {
                   return info.node;
                 });

  // compute transformations of all objects in the group relative to the root,
  // which are the absolute transformations
  std::vector<Mn::Matrix4> absTransforms =
      scene->transformationMatrices(objects);

  return absTransforms;
}

void ResourceManager::translateMesh(BaseMesh* meshDataGL,
                                    Magnum::Vector3 translation) {
  CollisionMeshData& meshData = meshDataGL->getCollisionMeshData();

  Magnum::Matrix4 transform = Magnum::Matrix4::translation(translation);
  Magnum::MeshTools::transformPointsInPlace(transform, meshData.positions);
  // save the mesh transformation for future query
  meshDataGL->meshTransform_ = transform * meshDataGL->meshTransform_;

  meshDataGL->BB = meshDataGL->BB.translated(translation);
}

void ResourceManager::buildPrimitiveAssetData(
    AbstractPrimitiveAttributes::ptr primTemplate) {
  // check if unique name of attributes describing primitive asset is present
  // already - don't remake if so
  auto primAssetOriginHandle = primTemplate->getOriginHandle();
  if (resourceDict_.count(primAssetOriginHandle) > 0) {
    LOG(INFO) << " Primitive Asset exists already : " << primAssetOriginHandle;
    return;
  }

  // class of primitive object
  std::string primClassName = primTemplate->getPrimObjClassName();
  // configuration for PrimitiveImporter - replace appropriate group's data
  // before instancing prim object
  auto conf = primImporter_->configuration();
  auto cfgGroup = conf.group(primClassName);
  if (cfgGroup != nullptr) {  // ignore prims with no configuration like cubes
    auto newCfgGroup = primTemplate->getConfigGroup();
    // replace current conf group with passed attributes
    *cfgGroup = newCfgGroup;
  }

  // make assetInfo
  AssetInfo info{AssetType::PRIMITIVE};
  info.requiresLighting = true;
  // set up primitive mesh
  // make  primitive mesh structure
  auto primMeshData = std::make_unique<GenericMeshData>(false);
  // build mesh data object
  primMeshData->importAndSetMeshData(*primImporter_, primClassName);

  // compute the mesh bounding box
  primMeshData->BB = computeMeshBB(primMeshData.get());

  primMeshData->uploadBuffersToGPU(false);

  // make MeshMetaData
  int meshStart = meshes_.size();
  int meshEnd = meshStart;
  MeshMetaData meshMetaData{meshStart, meshEnd};

  meshes_.emplace_back(std::move(primMeshData));

  // default material for now
  std::unique_ptr<gfx::MaterialData> phongMaterial =
      gfx::PhongMaterialData::create_unique();

  meshMetaData.setMaterialIndices(nextMaterialID_, nextMaterialID_);
  shaderManager_.set(std::to_string(nextMaterialID_++),
                     phongMaterial.release());

  meshMetaData.root.meshIDLocal = 0;
  meshMetaData.root.componentID = 0;
  // store the rotation to world frame upon load - currently superfluous
  const quatf transform = info.frame.rotationFrameToWorld();
  Magnum::Matrix4 R = Magnum::Matrix4::from(
      Magnum::Quaternion(transform).toMatrix(), Magnum::Vector3());
  meshMetaData.root.transformFromLocalToParent =
      R * meshMetaData.root.transformFromLocalToParent;

  // make LoadedAssetData corresponding to this asset
  LoadedAssetData loadedAssetData{info, meshMetaData};
  auto inserted =
      resourceDict_.emplace(primAssetOriginHandle, std::move(loadedAssetData));

  LOG(INFO) << " Primitive Asset Added : ID : "
            << primTemplate->getAssetTemplateID()
            << " : attr lib key : " << primTemplate->getOriginHandle()
            << " | instance class : " << primClassName
            << " | Conf has group for this obj type : "
            << conf.hasGroup(primClassName);

}  // buildPrimitiveAssetData

bool ResourceManager::loadPTexMeshData(const AssetInfo& info,
                                       scene::SceneNode* parent,
                                       DrawableGroup* drawables) {
#ifdef ESP_BUILD_PTEX_SUPPORT
  // if this is a new file, load it and add it to the dictionary
  const std::string& filename = info.filepath;
  if (resourceDict_.count(filename) == 0) {
    const auto atlasDir = Corrade::Utility::Directory::join(
        Corrade::Utility::Directory::path(filename), "textures");

    meshes_.emplace_back(std::make_unique<PTexMeshData>());
    int index = meshes_.size() - 1;
    auto* pTexMeshData = dynamic_cast<PTexMeshData*>(meshes_[index].get());
    pTexMeshData->load(filename, atlasDir);

    // update the dictionary
    auto inserted =
        resourceDict_.emplace(filename, LoadedAssetData{info, {index, index}});
    MeshMetaData& meshMetaData = inserted.first->second.meshMetaData;
    meshMetaData.root.meshIDLocal = 0;
    meshMetaData.root.componentID = 0;
    // store the rotation to world frame upon load
    const quatf transform = info.frame.rotationFrameToWorld();
    Magnum::Matrix4 R = Magnum::Matrix4::from(
        Magnum::Quaternion(transform).toMatrix(), Magnum::Vector3());
    meshMetaData.root.transformFromLocalToParent =
        R * meshMetaData.root.transformFromLocalToParent;
  }

  // create the scene graph by request
  if (parent) {
    auto indexPair = getMeshMetaData(filename).meshIndex;
    int start = indexPair.first;
    int end = indexPair.second;

    for (int iMesh = start; iMesh <= end; ++iMesh) {
      auto* pTexMeshData = dynamic_cast<PTexMeshData*>(meshes_[iMesh].get());

      pTexMeshData->uploadBuffersToGPU(false);

      for (int jSubmesh = 0; jSubmesh < pTexMeshData->getSize(); ++jSubmesh) {
        scene::SceneNode& node = parent->createChild();
        const quatf transform = info.frame.rotationFrameToWorld();
        node.setRotation(Magnum::Quaternion(transform));

        node.addFeature<gfx::PTexMeshDrawable>(*pTexMeshData, jSubmesh,
                                               shaderManager_, drawables);

        if (computeAbsoluteAABBs_) {
          staticDrawableInfo_.emplace_back(
              StaticDrawableInfo{node, static_cast<uint32_t>(jSubmesh)});
        }
      }
    }
  }

  return true;
#else
  LOG(ERROR) << "PTex support not enabled. Enable the BUILD_PTEX_SUPPORT CMake "
                "option when building.";
  return false;
#endif
}

// semantic instance mesh import
bool ResourceManager::loadInstanceMeshData(
    const AssetInfo& info,
    scene::SceneNode* parent,
    DrawableGroup* drawables,
    bool splitSemanticMesh /* = true */) {
  if (info.type != AssetType::INSTANCE_MESH) {
    LOG(ERROR) << "loadInstanceMeshData only works with INSTANCE_MESH type!";
    return false;
  }

  Cr::Containers::Pointer<Importer> importer;
  CORRADE_INTERNAL_ASSERT_OUTPUT(
      importer = importerManager_.loadAndInstantiate("StanfordImporter"));

  // if this is a new file, load it and add it to the dictionary, create
  // shaders and add it to the shaderPrograms_
  const std::string& filename = info.filepath;
  if (resourceDict_.count(filename) == 0) {
    std::vector<GenericInstanceMeshData::uptr> instanceMeshes;
    if (splitSemanticMesh) {
      instanceMeshes =
          GenericInstanceMeshData::fromPlySplitByObjectId(*importer, filename);
    } else {
      GenericInstanceMeshData::uptr meshData =
          GenericInstanceMeshData::fromPLY(*importer, filename);
      if (meshData)
        instanceMeshes.emplace_back(std::move(meshData));
    }

    if (instanceMeshes.empty()) {
      LOG(ERROR) << "Error loading instance mesh data";
      return false;
    }

    int meshStart = meshes_.size();
    int meshEnd = meshStart + instanceMeshes.size() - 1;
    MeshMetaData meshMetaData{meshStart, meshEnd};
    meshMetaData.root.children.resize(instanceMeshes.size());

    for (int meshIDLocal = 0; meshIDLocal < instanceMeshes.size();
         ++meshIDLocal) {
      instanceMeshes[meshIDLocal]->uploadBuffersToGPU(false);
      meshes_.emplace_back(std::move(instanceMeshes[meshIDLocal]));

      meshMetaData.root.children[meshIDLocal].meshIDLocal = meshIDLocal;
    }

    // update the dictionary
    resourceDict_.emplace(filename,
                          LoadedAssetData{info, std::move(meshMetaData)});
  }

  // create the scene graph by request
  if (parent) {
    auto indexPair = getMeshMetaData(filename).meshIndex;
    int start = indexPair.first;
    int end = indexPair.second;

    for (uint32_t iMesh = start; iMesh <= end; ++iMesh) {
      scene::SceneNode& node = parent->createChild();
      node.addFeature<gfx::GenericDrawable>(
          *meshes_[iMesh]->getMagnumGLMesh(), shaderManager_, NO_LIGHT_KEY,
          PER_VERTEX_OBJECT_ID_MATERIAL_KEY, drawables);

      if (computeAbsoluteAABBs_) {
        staticDrawableInfo_.emplace_back(StaticDrawableInfo{node, iMesh});
      }
    }
  }

  return true;
}

bool ResourceManager::loadGeneralMeshData(
    const AssetInfo& info,
    scene::SceneNode* parent /* = nullptr */,
    DrawableGroup* drawables /* = nullptr */,
    const Mn::ResourceKey& lightSetup) {
  const std::string& filename = info.filepath;
  const bool fileIsLoaded = resourceDict_.count(filename) > 0;
  const bool drawData = parent != nullptr && drawables != nullptr;

  Cr::Containers::Pointer<Importer> importer;
  CORRADE_INTERNAL_ASSERT_OUTPUT(
      importer = importerManager_.loadAndInstantiate("AnySceneImporter"));

  // Preferred plugins, Basis target GPU format
  importerManager_.setPreferredPlugins("GltfImporter", {"TinyGltfImporter"});
#ifdef ESP_BUILD_ASSIMP_SUPPORT
  importerManager_.setPreferredPlugins("ObjImporter", {"AssimpImporter"});
#endif
  {
    Cr::PluginManager::PluginMetadata* const metadata =
        importerManager_.metadata("BasisImporter");
    Mn::GL::Context& context = Mn::GL::Context::current();
#ifdef MAGNUM_TARGET_WEBGL
    if (context.isExtensionSupported<
            Mn::GL::Extensions::WEBGL::compressed_texture_astc>())
#else
    if (context.isExtensionSupported<
            Mn::GL::Extensions::KHR::texture_compression_astc_ldr>())
#endif
    {
      LOG(INFO) << "Importing Basis files as ASTC 4x4";
      metadata->configuration().setValue("format", "Astc4x4RGBA");
    }
#ifdef MAGNUM_TARGET_GLES
    else if (context.isExtensionSupported<
                 Mn::GL::Extensions::EXT::texture_compression_bptc>())
#else
    else if (context.isExtensionSupported<
                 Mn::GL::Extensions::ARB::texture_compression_bptc>())
#endif
    {
      LOG(INFO) << "Importing Basis files as BC7";
      metadata->configuration().setValue("format", "Bc7RGBA");
    }
#ifdef MAGNUM_TARGET_WEBGL
    else if (context.isExtensionSupported<
                 Mn::GL::Extensions::WEBGL::compressed_texture_s3tc>())
#elif defined(MAGNUM_TARGET_GLES)
    else if (context.isExtensionSupported<
                 Mn::GL::Extensions::EXT::texture_compression_s3tc>() ||
             context.isExtensionSupported<
                 Mn::GL::Extensions::ANGLE::texture_compression_dxt5>())
#else
    else if (context.isExtensionSupported<
                 Mn::GL::Extensions::EXT::texture_compression_s3tc>())
#endif
    {
      LOG(INFO) << "Importing Basis files as BC3";
      metadata->configuration().setValue("format", "Bc3RGBA");
    }
#ifndef MAGNUM_TARGET_GLES2
    else
#ifndef MAGNUM_TARGET_GLES
        if (context.isExtensionSupported<
                Mn::GL::Extensions::ARB::ES3_compatibility>())
#endif
    {
      LOG(INFO) << "Importing Basis files as ETC2";
      metadata->configuration().setValue("format", "Etc2RGBA");
    }
#else /* For ES2, fall back to PVRTC as ETC2 is not available */
    else
#ifdef MAGNUM_TARGET_WEBGL
        if (context.isExtensionSupported<Mn::WEBGL::compressed_texture_pvrtc>())
#else
        if (context.isExtensionSupported<Mn::IMG::texture_compression_pvrtc>())
#endif
    {
      LOG(INFO) << "Importing Basis files as PVRTC 4bpp";
      metadata->configuration().setValue("format", "PvrtcRGBA4bpp");
    }
#endif
#if defined(MAGNUM_TARGET_GLES2) || !defined(MAGNUM_TARGET_GLES)
    else /* ES3 has ETC2 always */
    {
      LOG(WARNING) << "No supported GPU compressed texture format detected, "
                      "Basis images will get imported as RGBA8";
      metadata->configuration().setValue("format", "RGBA8");
    }
#endif
  }

  // Optional File loading
  if (!fileIsLoaded) {
    if (!importer->openFile(filename)) {
      LOG(ERROR) << "Cannot open file " << filename;
      return false;
    }

    // if this is a new file, load it and add it to the dictionary
    LoadedAssetData loadedAssetData{info};
    loadTextures(*importer, loadedAssetData);
    loadMaterials(*importer, loadedAssetData);
    loadMeshes(*importer, loadedAssetData);
    auto inserted = resourceDict_.emplace(filename, std::move(loadedAssetData));
    MeshMetaData& meshMetaData = inserted.first->second.meshMetaData;

    // Register magnum mesh
    if (importer->defaultScene() != -1) {
      Corrade::Containers::Optional<Magnum::Trade::SceneData> sceneData =
          importer->scene(importer->defaultScene());
      if (!sceneData) {
        LOG(ERROR) << "Cannot load scene, exiting";
        return false;
      }
      for (unsigned int sceneDataID : sceneData->children3D()) {
        loadMeshHierarchy(*importer, meshMetaData.root, sceneDataID);
      }
    } else if (importer->meshCount() && meshes_[meshMetaData.meshIndex.first]) {
      // no default scene --- standalone OBJ/PLY files, for example
      // take a wild guess and load the first mesh with the first material
      // addMeshToDrawables(metaData, *parent, drawables, ID_UNDEFINED, 0, 0);
      loadMeshHierarchy(*importer, meshMetaData.root, 0);
    } else {
      LOG(ERROR) << "No default scene available and no meshes found, exiting";
      return false;
    }

    const quatf transform = info.frame.rotationFrameToWorld();
    Magnum::Matrix4 R = Magnum::Matrix4::from(
        Magnum::Quaternion(transform).toMatrix(), Magnum::Vector3());
    meshMetaData.root.transformFromLocalToParent =
        R * meshMetaData.root.transformFromLocalToParent;
  } else if (resourceDict_[filename].assetInfo != info) {
    // Right now, we only allow for an asset to be loaded with one
    // configuration, since generated mesh data may be invalid for a new
    // configuration
    LOG(ERROR) << "Reloading asset " << filename
               << " with different configuration not currently supported. "
               << "Asset may not be rendered correctly.";
  }

  // Optional Instantiation
  if (!drawData) {
    //! Do not instantiate object
    return true;
  }

  //! Do instantiate object
  const LoadedAssetData& loadedAssetData = resourceDict_[filename];
  if (!isLightSetupCompatible(loadedAssetData, lightSetup)) {
    LOG(WARNING) << "Loading scene with incompatible light setup, "
                    "scene will not be correctly lit. If the scene requires "
                    "lighting please enable AssetInfo::requiresLighting.";
  }
  const MeshMetaData& meshMetaData = loadedAssetData.meshMetaData;

  scene::SceneNode& newNode = parent->createChild();
  const bool forceReload = false;
  // re-bind position, normals, uv, colors etc. to the corresponding buffers
  // under *current* gl context
  if (forceReload) {
    int start = meshMetaData.meshIndex.first;
    int end = meshMetaData.meshIndex.second;
    if (0 <= start && start <= end) {
      for (int iMesh = start; iMesh <= end; ++iMesh) {
        meshes_[iMesh]->uploadBuffersToGPU(forceReload);
      }
    }
  }  // forceReload

  addComponent(meshMetaData, newNode, lightSetup, drawables, meshMetaData.root);
  return true;
}  // loadGeneralMeshData

int ResourceManager::loadNavMeshVisualization(esp::nav::PathFinder& pathFinder,
                                              scene::SceneNode* parent,
                                              DrawableGroup* drawables) {
  int navMeshPrimitiveID = ID_UNDEFINED;

  if (!pathFinder.isLoaded())
    return navMeshPrimitiveID;

  // create the mesh
  std::vector<Magnum::UnsignedInt> indices;
  std::vector<Magnum::Vector3> positions;

  const MeshData::ptr navMeshData = pathFinder.getNavMeshData();

  // add the vertices
  positions.resize(navMeshData->vbo.size());
  for (size_t vix = 0; vix < navMeshData->vbo.size(); vix++) {
    positions[vix] = Magnum::Vector3{navMeshData->vbo[vix]};
  }

  indices.resize(navMeshData->ibo.size() * 2);
  for (size_t ix = 0; ix < navMeshData->ibo.size();
       ix += 3) {  // for each triangle, create lines
    size_t nix = ix * 2;
    indices[nix] = navMeshData->ibo[ix];
    indices[nix + 1] = navMeshData->ibo[ix + 1];
    indices[nix + 2] = navMeshData->ibo[ix + 1];
    indices[nix + 3] = navMeshData->ibo[ix + 2];
    indices[nix + 4] = navMeshData->ibo[ix + 2];
    indices[nix + 5] = navMeshData->ibo[ix];
  }

  // create a temporary mesh object referencing the above data
  Mn::Trade::MeshData visualNavMesh{
      Mn::MeshPrimitive::Lines,
      {},
      indices,
      Mn::Trade::MeshIndexData{indices},
      {},
      positions,
      {Mn::Trade::MeshAttributeData{Mn::Trade::MeshAttribute::Position,
                                    Cr::Containers::arrayView(positions)}}};

  // compile and add the new mesh to the structure
  primitive_meshes_.push_back(std::make_unique<Magnum::GL::Mesh>(
      Magnum::MeshTools::compile(visualNavMesh)));

  navMeshPrimitiveID = primitive_meshes_.size() - 1;

  if (parent != nullptr && drawables != nullptr &&
      navMeshPrimitiveID != ID_UNDEFINED) {
    // create the drawable
    addPrimitiveToDrawables(navMeshPrimitiveID, *parent, drawables);
  }

  return navMeshPrimitiveID;
}  // loadNavMeshVisualization

void ResourceManager::loadMaterials(Importer& importer,
                                    LoadedAssetData& loadedAssetData) {
  int materialStart = nextMaterialID_;
  int materialEnd = materialStart + importer.materialCount() - 1;
  loadedAssetData.meshMetaData.setMaterialIndices(materialStart, materialEnd);

  for (int iMaterial = 0; iMaterial < importer.materialCount(); ++iMaterial) {
    int currentMaterialID = nextMaterialID_++;

    // TODO:
    // it seems we have a way to just load the material once in this case,
    // as long as the materialName includes the full path to the material
    std::unique_ptr<Magnum::Trade::AbstractMaterialData> materialData =
        importer.material(iMaterial);
    if (!materialData ||
        materialData->type() != Magnum::Trade::MaterialType::Phong) {
      LOG(ERROR) << "Cannot load material, skipping";
      continue;
    }

    const auto& phongMaterialData =
        static_cast<Mn::Trade::PhongMaterialData&>(*materialData);
    std::unique_ptr<gfx::MaterialData> finalMaterial;
    int textureBaseIndex = loadedAssetData.meshMetaData.textureIndex.first;
    if (loadedAssetData.assetInfo.requiresLighting) {
      finalMaterial =
          buildPhongShadedMaterialData(phongMaterialData, textureBaseIndex);

    } else {
      finalMaterial =
          buildFlatShadedMaterialData(phongMaterialData, textureBaseIndex);
    }
    // for now, just use unique ID for material key. This may change if we
    // expose materials to user for post-load modification
    shaderManager_.set(std::to_string(currentMaterialID),
                       finalMaterial.release());
  }
}

gfx::PhongMaterialData::uptr ResourceManager::buildFlatShadedMaterialData(
    const Mn::Trade::PhongMaterialData& material,
    int textureBaseIndex) {
  // NOLINTNEXTLINE(google-build-using-namespace)
  using namespace Mn::Math::Literals;

  auto finalMaterial = gfx::PhongMaterialData::create_unique();
  finalMaterial->ambientColor = 0xffffffff_rgbaf;
  finalMaterial->diffuseColor = 0x00000000_rgbaf;
  finalMaterial->specularColor = 0x00000000_rgbaf;

  if (material.flags() & Mn::Trade::PhongMaterialData::Flag::AmbientTexture) {
    finalMaterial->ambientTexture =
        textures_[textureBaseIndex + material.ambientTexture()].get();
  } else if (material.flags() &
             Mn::Trade::PhongMaterialData::Flag::DiffuseTexture) {
    // if we want to force flat shading, but we don't have ambient texture,
    // check for diffuse texture and use that instead
    finalMaterial->ambientTexture =
        textures_[textureBaseIndex + material.diffuseTexture()].get();
  } else {
    finalMaterial->ambientColor = material.ambientColor();
  }
  return finalMaterial;
}

gfx::PhongMaterialData::uptr ResourceManager::buildPhongShadedMaterialData(
    const Mn::Trade::PhongMaterialData& material,
    int textureBaseIndex) {
  // NOLINTNEXTLINE(google-build-using-namespace)
  using namespace Mn::Math::Literals;

  auto finalMaterial = gfx::PhongMaterialData::create_unique();
  finalMaterial->shininess = material.shininess();

  // texture transform, if there's none the matrix is an identity
  finalMaterial->textureMatrix = material.textureMatrix();

  // ambient material properties
  finalMaterial->ambientColor = material.ambientColor();
  if (material.flags() & Mn::Trade::PhongMaterialData::Flag::AmbientTexture) {
    finalMaterial->ambientTexture =
        textures_[textureBaseIndex + material.ambientTexture()].get();
  }

  // diffuse material properties
  finalMaterial->diffuseColor = material.diffuseColor();
  if (material.flags() & Mn::Trade::PhongMaterialData::Flag::DiffuseTexture) {
    finalMaterial->diffuseTexture =
        textures_[textureBaseIndex + material.diffuseTexture()].get();
  }

  // specular material properties
  finalMaterial->specularColor = material.specularColor();
  if (material.flags() & Mn::Trade::PhongMaterialData::Flag::SpecularTexture) {
    finalMaterial->specularTexture =
        textures_[textureBaseIndex + material.specularTexture()].get();
  }

  // normal mapping
  if (material.flags() & Mn::Trade::PhongMaterialData::Flag::NormalTexture) {
    finalMaterial->normalTexture =
        textures_[textureBaseIndex + material.normalTexture()].get();
  }
  return finalMaterial;
}

void ResourceManager::loadMeshes(Importer& importer,
                                 LoadedAssetData& loadedAssetData) {
  int meshStart = meshes_.size();
  int meshEnd = meshStart + importer.meshCount() - 1;
  loadedAssetData.meshMetaData.setMeshIndices(meshStart, meshEnd);

  for (int iMesh = 0; iMesh < importer.meshCount(); ++iMesh) {
    // don't need normals if we aren't using lighting
    auto gltfMeshData = std::make_unique<GenericMeshData>(
        loadedAssetData.assetInfo.requiresLighting);
    gltfMeshData->importAndSetMeshData(importer, iMesh);

    // compute the mesh bounding box
    gltfMeshData->BB = computeMeshBB(gltfMeshData.get());

    gltfMeshData->uploadBuffersToGPU(false);
    meshes_.emplace_back(std::move(gltfMeshData));
  }
}

//! Recursively load the transformation chain specified by the mesh file
void ResourceManager::loadMeshHierarchy(Importer& importer,
                                        MeshTransformNode& parent,
                                        int componentID) {
  std::unique_ptr<Magnum::Trade::ObjectData3D> objectData =
      importer.object3D(componentID);
  if (!objectData) {
    LOG(ERROR) << "Cannot import object " << importer.object3DName(componentID)
               << ", skipping";
    return;
  }

  // Add the new node to the hierarchy and set its transformation
  parent.children.push_back(MeshTransformNode());
  parent.children.back().transformFromLocalToParent =
      objectData->transformation();
  parent.children.back().componentID = componentID;

  const int meshIDLocal = objectData->instance();

  // Add a mesh index
  if (objectData->instanceType() == Magnum::Trade::ObjectInstanceType3D::Mesh &&
      meshIDLocal != ID_UNDEFINED) {
    parent.children.back().meshIDLocal = meshIDLocal;
    parent.children.back().materialIDLocal =
        static_cast<Magnum::Trade::MeshObjectData3D*>(objectData.get())
            ->material();
  }

  // Recursively add children
  for (auto childObjectID : objectData->children()) {
    loadMeshHierarchy(importer, parent.children.back(), childObjectID);
  }
}

void ResourceManager::loadTextures(Importer& importer,
                                   LoadedAssetData& loadedAssetData) {
  int textureStart = textures_.size();
  int textureEnd = textureStart + importer.textureCount() - 1;
  loadedAssetData.meshMetaData.setTextureIndices(textureStart, textureEnd);

  for (int iTexture = 0; iTexture < importer.textureCount(); ++iTexture) {
    textures_.emplace_back(std::make_shared<Magnum::GL::Texture2D>());
    auto& currentTexture = textures_.back();

    auto textureData = importer.texture(iTexture);
    if (!textureData ||
        textureData->type() != Magnum::Trade::TextureData::Type::Texture2D) {
      LOG(ERROR) << "Cannot load texture " << iTexture << " skipping";
      currentTexture = nullptr;
      continue;
    }

    // Configure the texture
    Mn::GL::Texture2D& texture = *(textures_[textureStart + iTexture].get());
    texture.setMagnificationFilter(textureData->magnificationFilter())
        .setMinificationFilter(textureData->minificationFilter(),
                               textureData->mipmapFilter())
        .setWrapping(textureData->wrapping().xy());

    // Load all mip levels
    const std::uint32_t levelCount =
        importer.image2DLevelCount(textureData->image());
    bool generateMipmap = false;
    for (std::uint32_t level = 0; level != levelCount; ++level) {
      // TODO:
      // it seems we have a way to just load the image once in this case,
      // as long as the image2DName include the full path to the image
      Cr::Containers::Optional<Mn::Trade::ImageData2D> image =
          importer.image2D(textureData->image(), level);
      if (!image) {
        LOG(ERROR) << "Cannot load texture image, skipping";
        currentTexture = nullptr;
        break;
      }

      Mn::GL::TextureFormat format;
      if (image->isCompressed()) {
        format = Mn::GL::textureFormat(image->compressedFormat());
      } else if (compressTextures_ &&
                 image->format() == Mn::PixelFormat::RGBA8Unorm) {
        format = Mn::GL::TextureFormat::CompressedRGBAS3tcDxt1;
      } else if (compressTextures_ &&
                 image->format() == Mn::PixelFormat::RGB8Unorm) {
        format = Mn::GL::TextureFormat::CompressedRGBS3tcDxt1;
      } else {
        format = Mn::GL::textureFormat(image->format());
      }

      // For the very first level, allocate the texture
      if (level == 0) {
        // If there is just one level and the image is not compressed, we'll
        // generate mips ourselves
        if (levelCount == 1 && !image->isCompressed()) {
          texture.setStorage(Mn::Math::log2(image->size().max()) + 1, format,
                             image->size());
          generateMipmap = true;
        } else
          texture.setStorage(levelCount, format, image->size());
      }

      if (image->isCompressed())
        texture.setCompressedSubImage(level, {}, *image);
      else
        texture.setSubImage(level, {}, *image);
    }

    // Mip level loading failed, fail the whole texture
    if (currentTexture == nullptr)
      continue;

    // Generate a mipmap if requested
    if (generateMipmap)
      texture.generateMipmap();
  }
}

void ResourceManager::addObjectToDrawables(const std::string& objTemplateHandle,
                                           scene::SceneNode* parent,
                                           DrawableGroup* drawables,
                                           const Mn::ResourceKey& lightSetup) {
  if (parent != nullptr and drawables != nullptr) {
    //! Add mesh to rendering stack

    // Meta data
    PhysicsObjectAttributes::ptr physicsObjectAttributes =
        physicsObjTemplateLibrary_.at(objTemplateHandle);

    const std::string& renderObjectName =
        physicsObjectAttributes->getRenderAssetHandle();
    // if no assets have been registered with resourceDict then do so
    // should only happen with primitives, since all file-based resources should
    // be loaded by now
    if (resourceDict_.count(renderObjectName) == 0) {
      // needs to have a primitive asset attributes with same name
      if (primitiveAssetsTemplateLibrary_.count(renderObjectName) == 0) {
        // this is bad, means no render primitive template exists with expected
        // name.  should never happen
        LOG(ERROR) << "No primitive asset attributes exists with name :"
                   << renderObjectName
                   << " so unable to instantiate primitive-based render "
                      "object.  Aborting.";
        return;
      }
      // build primitive asset for this object based on defined primitive
      // attributes
      auto primitiveAssetAttributes =
          primitiveAssetsTemplateLibrary_.at(renderObjectName);
      buildPrimitiveAssetData(primitiveAssetAttributes);
    }
    const LoadedAssetData& loadedAssetData = resourceDict_.at(renderObjectName);
    if (!isLightSetupCompatible(loadedAssetData, lightSetup)) {
      LOG(WARNING) << "Instantiating object with incompatible light setup, "
                      "object will not be correctly lit. If you need lighting "
                      "please ensure 'requires lighting' is enabled in object "
                      "config file";
    }

    // need a new node for scaling because motion state will override scale
    // set at the physical node
    scene::SceneNode& scalingNode = parent->createChild();
    Magnum::Vector3 objectScaling = physicsObjectAttributes->getScale();
    scalingNode.setScaling(objectScaling);

    addComponent(loadedAssetData.meshMetaData, scalingNode, lightSetup,
                 drawables, loadedAssetData.meshMetaData.root);
  }  // should always be specified, otherwise won't do anything
}  // addObjectToDrawables

//! Add component to rendering stack, based on importer loading
void ResourceManager::addComponent(const MeshMetaData& metaData,
                                   scene::SceneNode& parent,
                                   const Mn::ResourceKey& lightSetup,
                                   DrawableGroup* drawables,
                                   const MeshTransformNode& meshTransformNode) {
  // Add the object to the scene and set its transformation
  scene::SceneNode& node = parent.createChild();
  node.MagnumObject::setTransformation(
      meshTransformNode.transformFromLocalToParent);

  const int meshIDLocal = meshTransformNode.meshIDLocal;

  // Add a drawable if the object has a mesh and the mesh is loaded
  if (meshIDLocal != ID_UNDEFINED) {
    const int materialIDLocal = meshTransformNode.materialIDLocal;
    addMeshToDrawables(metaData, node, lightSetup, drawables,
                       meshTransformNode.componentID, meshIDLocal,
                       materialIDLocal);

    // compute the bounding box for the mesh we are adding
    const int meshID = metaData.meshIndex.first + meshIDLocal;
    BaseMesh* mesh = meshes_[meshID].get();
    node.setMeshBB(computeMeshBB(mesh));
  }

  // Recursively add children
  for (auto& child : meshTransformNode.children) {
    addComponent(metaData, node, lightSetup, drawables, child);
  }
}  // addComponent

void ResourceManager::addMeshToDrawables(const MeshMetaData& metaData,
                                         scene::SceneNode& node,
                                         const Mn::ResourceKey& lightSetup,
                                         DrawableGroup* drawables,
                                         int objectID,
                                         int meshIDLocal,
                                         int materialIDLocal) {
  const int meshStart = metaData.meshIndex.first;
  const uint32_t meshID = meshStart + meshIDLocal;
  Magnum::GL::Mesh& mesh = *meshes_[meshID]->getMagnumGLMesh();

  Mn::ResourceKey materialKey;
  if (materialIDLocal == ID_UNDEFINED ||
      metaData.materialIndex.second == ID_UNDEFINED) {
    materialKey = DEFAULT_MATERIAL_KEY;
  } else {
    materialKey =
        std::to_string(metaData.materialIndex.first + materialIDLocal);
  }

  createGenericDrawable(mesh, node, lightSetup, materialKey, drawables,
                        objectID);

  if (computeAbsoluteAABBs_) {
    staticDrawableInfo_.emplace_back(StaticDrawableInfo{node, meshID});
  }
}  // addMeshToDrawables

void ResourceManager::addPrimitiveToDrawables(int primitiveID,
                                              scene::SceneNode& node,
                                              DrawableGroup* drawables) {
  CHECK(primitiveID >= 0 && primitiveID < primitive_meshes_.size());
  createGenericDrawable(*primitive_meshes_[primitiveID], node,
                        DEFAULT_LIGHTING_KEY, DEFAULT_MATERIAL_KEY, drawables);
}

void ResourceManager::createGenericDrawable(
    Mn::GL::Mesh& mesh,
    scene::SceneNode& node,
    const Mn::ResourceKey& lightSetup,
    const Mn::ResourceKey& material,
    DrawableGroup* group /* = nullptr */,
    int objectId /* = ID_UNDEFINED */) {
  node.addFeature<gfx::GenericDrawable>(mesh, shaderManager_, lightSetup,
                                        material, group, objectId);
}

bool ResourceManager::loadSUNCGHouseFile(const AssetInfo& houseInfo,
                                         scene::SceneNode* parent,
                                         DrawableGroup* drawables) {
  ASSERT(parent != nullptr);
  std::string houseFile = Cr::Utility::Directory::join(
      Cr::Utility::Directory::current(), houseInfo.filepath);
  const auto& json = io::parseJsonFile(houseFile);
  const auto& levels = json["levels"].GetArray();
  std::vector<std::string> pathTokens = io::tokenize(houseFile, "/", 0, true);
  ASSERT(pathTokens.size() >= 3);
  pathTokens.pop_back();  // house.json
  const std::string houseId = pathTokens.back();
  pathTokens.pop_back();  // <houseId>
  pathTokens.pop_back();  // house
  const std::string basePath = Corrade::Utility::String::join(pathTokens, '/');

  // store nodeIds to obtain linearized index for semantic masks
  std::vector<std::string> nodeIds;

  for (const auto& level : levels) {
    const auto& nodes = level["nodes"].GetArray();
    for (const auto& node : nodes) {
      const std::string nodeId = node["id"].GetString();
      const std::string nodeType = node["type"].GetString();
      const int valid = node["valid"].GetInt();
      if (valid == 0) {
        continue;
      }

      // helper for creating object nodes
      auto createObjectFunc = [&](const AssetInfo& info,
                                  const std::string& id) -> scene::SceneNode& {
        scene::SceneNode& objectNode = parent->createChild();
        const int nodeIndex = nodeIds.size();
        nodeIds.push_back(id);
        objectNode.setId(nodeIndex);
        if (info.type == AssetType::SUNCG_OBJECT) {
          loadGeneralMeshData(info, &objectNode, drawables);
        }
        return objectNode;
      };

      const std::string roomPath = basePath + "/room/" + houseId + "/";
      if (nodeType == "Room") {
        const std::string roomBase = roomPath + node["modelId"].GetString();
        const int hideCeiling = node["hideCeiling"].GetInt();
        const int hideFloor = node["hideFloor"].GetInt();
        const int hideWalls = node["hideWalls"].GetInt();
        if (hideCeiling != 1) {
          createObjectFunc({AssetType::SUNCG_OBJECT, roomBase + "c.glb"},
                           nodeId + "c");
        }
        if (hideWalls != 1) {
          createObjectFunc({AssetType::SUNCG_OBJECT, roomBase + "w.glb"},
                           nodeId + "w");
        }
        if (hideFloor != 1) {
          createObjectFunc({AssetType::SUNCG_OBJECT, roomBase + "f.glb"},
                           nodeId + "f");
        }
      } else if (nodeType == "Object") {
        const std::string modelId = node["modelId"].GetString();
        // Parse model-to-scene transformation matrix
        // NOTE: only "Object" nodes have transform, other nodes are directly
        // specified in scene coordinates
        std::vector<float> transformVec;
        io::toFloatVector(node["transform"], &transformVec);
        mat4f transform(transformVec.data());
        const AssetInfo info{
            AssetType::SUNCG_OBJECT,
            basePath + "/object/" + modelId + "/" + modelId + ".glb"};
        createObjectFunc(info, nodeId)
            .setTransformation(Magnum::Matrix4{transform});
      } else if (nodeType == "Box") {
        // TODO(MS): create Box geometry
        createObjectFunc({}, nodeId);
      } else if (nodeType == "Ground") {
        const std::string roomBase = roomPath + node["modelId"].GetString();
        const AssetInfo info{AssetType::SUNCG_OBJECT, roomBase + "f.glb"};
        createObjectFunc(info, nodeId);
      } else {
        LOG(ERROR) << "Unrecognized SUNCG house node type " << nodeType;
      }
    }
  }
  return true;
}

void ResourceManager::initDefaultLightSetups() {
  shaderManager_.set(NO_LIGHT_KEY, gfx::LightSetup{});
  shaderManager_.setFallback(gfx::LightSetup{});
}

void ResourceManager::initDefaultMaterials() {
  shaderManager_.set<gfx::MaterialData>(DEFAULT_MATERIAL_KEY,
                                        new gfx::PhongMaterialData{});
  auto perVertexObjectId = new gfx::PhongMaterialData{};
  perVertexObjectId->perVertexObjectId = true;
  perVertexObjectId->vertexColored = true;
  perVertexObjectId->ambientColor = Mn::Color4{1.0};
  shaderManager_.set<gfx::MaterialData>(PER_VERTEX_OBJECT_ID_MATERIAL_KEY,
                                        perVertexObjectId);
  shaderManager_.setFallback<gfx::MaterialData>(new gfx::PhongMaterialData{});
}

bool ResourceManager::isLightSetupCompatible(
    const LoadedAssetData& loadedAssetData,
    const Magnum::ResourceKey& lightSetup) const {
  // if light setup has lights in it, but asset was loaded in as flat shaded,
  // there may be an error when rendering.
  return lightSetup == Mn::ResourceKey{NO_LIGHT_KEY} ||
         loadedAssetData.assetInfo.requiresLighting;
}

//! recursively join all sub-components of a mesh into a single unified
//! MeshData.
void ResourceManager::joinHeirarchy(
    MeshData& mesh,
    const MeshMetaData& metaData,
    const MeshTransformNode& node,
    const Magnum::Matrix4& transformFromParentToWorld) {
  Magnum::Matrix4 transformFromLocalToWorld =
      transformFromParentToWorld * node.transformFromLocalToParent;

  if (node.meshIDLocal != ID_UNDEFINED) {
    CollisionMeshData& meshData =
        meshes_[node.meshIDLocal + metaData.meshIndex.first]
            ->getCollisionMeshData();
    int lastIndex = mesh.vbo.size();
    for (auto& pos : meshData.positions) {
      mesh.vbo.push_back(Magnum::EigenIntegration::cast<vec3f>(
          transformFromLocalToWorld.transformPoint(pos)));
    }
    for (auto& index : meshData.indices) {
      mesh.ibo.push_back(index + lastIndex);
    }
  }

  for (auto& child : node.children) {
    joinHeirarchy(mesh, metaData, child, transformFromLocalToWorld);
  }
}

std::unique_ptr<MeshData> ResourceManager::createJoinedCollisionMesh(
    const std::string& filename) {
  std::unique_ptr<MeshData> mesh = std::make_unique<MeshData>();

  CHECK(resourceDict_.count(filename) > 0);

  const MeshMetaData& metaData = getMeshMetaData(filename);

  Magnum::Matrix4 identity;
  joinHeirarchy(*mesh, metaData, metaData.root, identity);

  return mesh;
}

}  // namespace assets
}  // namespace esp
