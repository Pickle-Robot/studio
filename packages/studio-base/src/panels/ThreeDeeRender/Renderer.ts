// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import EventEmitter from "eventemitter3";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer";
import { OutlinePass } from "three/examples/jsm/postprocessing/OutlinePass";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass";
import { GammaCorrectionShader } from "three/examples/jsm/shaders/GammaCorrectionShader";

import Logger from "@foxglove/log";
import { CameraState } from "@foxglove/regl-worldview";

import { Input } from "./Input";
import { LayerErrors } from "./LayerErrors";
import { MaterialCache } from "./MaterialCache";
import { ModelCache } from "./ModelCache";
import { Picker } from "./Picker";
import { DetailLevel } from "./lod";
import { FrameAxes } from "./renderables/FrameAxes";
import { Markers } from "./renderables/Markers";
import { OccupancyGrids } from "./renderables/OccupancyGrids";
import { PointClouds } from "./renderables/PointClouds";
import { Marker, OccupancyGrid, PointCloud2, TF } from "./ros";
import { TransformTree } from "./transforms/TransformTree";

const log = Logger.getLogger(__filename);

export type RendererEvents = {
  startFrame: (currentTime: bigint, renderer: Renderer) => void;
  endFrame: (currentTime: bigint, renderer: Renderer) => void;
  cameraMove: (renderer: Renderer) => void;
  renderableSelected: (renderable: THREE.Object3D, renderer: Renderer) => void;
  transformTreeUpdated: (renderer: Renderer) => void;
  showLabel: (labelId: string, labelMarker: Marker, renderer: Renderer) => void;
  removeLabel: (labelId: string, renderer: Renderer) => void;
};

const LIGHT_BACKDROP = new THREE.Color(0xececec).convertSRGBToLinear();
const DARK_BACKDROP = new THREE.Color(0x121217).convertSRGBToLinear();

const LIGHT_OUTLINE = new THREE.Color(0x121217).convertSRGBToLinear();
const DARK_OUTLINE = new THREE.Color(0xececec).convertSRGBToLinear();

const LIGHT_HIGHLIGHT = new THREE.Color(0xffffff).convertSRGBToLinear();
const DARK_HIGHLIGHT = new THREE.Color(0xffffff).convertSRGBToLinear();

const TRANSFORM_STORAGE_TIME_NS = 60n * BigInt(1e9);

const UNIT_X = new THREE.Vector3(1, 0, 0);
const PI_2 = Math.PI / 2;

const tempVec = new THREE.Vector3();
const tempVec2 = new THREE.Vector2();
const tempSpherical = new THREE.Spherical();
const tempEuler = new THREE.Euler();

export class Renderer extends EventEmitter<RendererEvents> {
  canvas: HTMLCanvasElement;
  gl: THREE.WebGLRenderer;
  maxLod = DetailLevel.High;
  target: THREE.WebGLRenderTarget;
  composer: EffectComposer;
  outlinePass: OutlinePass;
  scene: THREE.Scene;
  dirLight: THREE.DirectionalLight;
  hemiLight: THREE.HemisphereLight;
  input: Input;
  camera: THREE.PerspectiveCamera;
  picker: Picker;
  materialCache = new MaterialCache();
  layerErrors = new LayerErrors();
  colorScheme: "dark" | "light" | undefined;
  modelCache: ModelCache;
  renderables = new Map<string, THREE.Object3D>();
  transformTree = new TransformTree(TRANSFORM_STORAGE_TIME_NS);
  currentTime: bigint | undefined;
  fixedFrameId: string | undefined;
  renderFrameId: string | undefined;

  frameAxes = new FrameAxes(this);
  occupancyGrids = new OccupancyGrids(this);
  pointClouds = new PointClouds(this);
  markers = new Markers(this);

  constructor(canvas: HTMLCanvasElement) {
    super();

    // NOTE: Global side effect
    THREE.Object3D.DefaultUp = new THREE.Vector3(0, 0, 1);

    this.canvas = canvas;
    this.gl = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
    });
    if (!this.gl.capabilities.isWebGL2) {
      throw new Error("WebGL2 is not supported");
    }
    this.gl.outputEncoding = THREE.LinearEncoding;
    this.gl.toneMapping = THREE.NoToneMapping;
    this.gl.autoClear = false;
    this.gl.info.autoReset = false;
    this.gl.shadowMap.enabled = false;
    this.gl.shadowMap.type = THREE.VSMShadowMap;
    this.gl.setPixelRatio(window.devicePixelRatio);

    let width = canvas.width;
    let height = canvas.height;
    if (canvas.parentElement) {
      width = canvas.parentElement.clientWidth;
      height = canvas.parentElement.clientHeight;
      this.gl.setSize(width, height);
    }

    this.modelCache = new ModelCache({ ignoreColladaUpAxis: true });

    this.scene = new THREE.Scene();
    this.scene.add(this.frameAxes);
    this.scene.add(this.occupancyGrids);
    this.scene.add(this.pointClouds);
    this.scene.add(this.markers);

    this.dirLight = new THREE.DirectionalLight();
    this.dirLight.position.set(1, 1, 1);
    this.dirLight.castShadow = true;

    this.dirLight.shadow.mapSize.width = 2048;
    this.dirLight.shadow.mapSize.height = 2048;
    this.dirLight.shadow.camera.near = 0.5;
    this.dirLight.shadow.camera.far = 500;
    this.dirLight.shadow.bias = -0.00001;

    this.hemiLight = new THREE.HemisphereLight(0xffffff, 0xffffff, 0.5);

    this.scene.add(this.dirLight);
    this.scene.add(this.hemiLight);

    this.input = new Input(canvas);
    this.input.on("resize", (size) => this.resizeHandler(size));
    this.input.on("click", (cursorCoords) => this.clickHandler(cursorCoords));

    const fov = 79;
    const near = 0.01; // 1cm
    const far = 10_000; // 10km
    this.camera = new THREE.PerspectiveCamera(fov, width / height, near, far);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(1, -3, 1);
    this.camera.lookAt(0, 0, 0);

    this.picker = new Picker(this.gl, this.scene, this.camera);

    // NOTE: Type definition workaround
    const maxSamples = (this.gl.capabilities as unknown as { maxSamples: number }).maxSamples;
    const samples = Math.min(8, maxSamples);
    // NOTE: Type definition workaround
    const targetOpts = { samples } as THREE.WebGLRenderTargetOptions;
    const size = this.gl.getDrawingBufferSize(tempVec2);

    // TODO(jhurliman): Use this when it actually works (upgrade three.js?)
    // this.target = new THREE.WebGLRenderTarget(size.width, size.height, targetOpts);
    this.target = new THREE.WebGLMultisampleRenderTarget(size.width, size.height, targetOpts);
    this.outlinePass = new OutlinePass(size, this.scene, this.camera);
    this.outlinePass.edgeStrength = 3;
    this.composer = new EffectComposer(this.gl, this.target);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(this.outlinePass);
    this.composer.addPass(new ShaderPass(GammaCorrectionShader));

    log.debug(`Initialized ${size.width}x${size.height} renderer (${maxSamples}x MSAA)`);

    this.animationFrame();
  }

  dispose(): void {
    this.removeAllListeners();
    this.picker.dispose();
    this.input.dispose();
    this.frameAxes.dispose();
    this.occupancyGrids.dispose();
    this.pointClouds.dispose();
    this.markers.dispose();
    this.target.dispose();
    this.gl.dispose();
  }

  setColorScheme(colorScheme: "dark" | "light"): void {
    log.debug(`Setting color scheme to "${colorScheme}"`);
    this.colorScheme = colorScheme;
    if (colorScheme === "dark") {
      this.gl.setClearColor(DARK_BACKDROP);
      this.materialCache.outlineMaterial.color.set(DARK_OUTLINE);
      this.materialCache.outlineMaterial.needsUpdate = true;
      this.outlinePass.visibleEdgeColor.set(DARK_HIGHLIGHT);
      this.outlinePass.hiddenEdgeColor.set(DARK_HIGHLIGHT);
    } else {
      this.gl.setClearColor(LIGHT_BACKDROP);
      this.materialCache.outlineMaterial.color.set(LIGHT_OUTLINE);
      this.materialCache.outlineMaterial.needsUpdate = true;
      this.outlinePass.visibleEdgeColor.set(LIGHT_HIGHLIGHT);
      this.outlinePass.hiddenEdgeColor.set(LIGHT_HIGHLIGHT);
    }
  }

  addTransformMessage(tf: TF): void {
    this.frameAxes.addTransformMessage(tf);
  }

  addOccupancyGridMessage(topic: string, occupancyGrid: OccupancyGrid): void {
    this.occupancyGrids.addOccupancyGridMessage(topic, occupancyGrid);
  }

  addPointCloud2Message(topic: string, pointCloud: PointCloud2): void {
    this.pointClouds.addPointCloud2Message(topic, pointCloud);
  }

  addMarkerMessage(topic: string, marker: Marker): void {
    this.markers.addMarkerMessage(topic, marker);
  }

  markerWorldPosition(markerId: string): Readonly<THREE.Vector3> | undefined {
    const renderable = this.renderables.get(markerId);
    if (!renderable) {
      return undefined;
    }

    tempVec.set(0, 0, 0);
    tempVec.applyMatrix4(renderable.matrixWorld);
    return tempVec;
  }

  // Callback handlers

  animationFrame = (): void => {
    if (this.currentTime != undefined) {
      this.frameHandler(this.currentTime);
    }
  };

  frameHandler = (currentTime: bigint): void => {
    this.emit("startFrame", currentTime, this);

    // TODO: Remove this hack when the user can set the renderFrameId themselves
    this.fixedFrameId = "map";
    this.renderFrameId = "base_link";

    this.materialCache.update(this.input.canvasSize);

    this.frameAxes.startFrame(currentTime);
    this.occupancyGrids.startFrame(currentTime);
    this.pointClouds.startFrame(currentTime);
    this.markers.startFrame(currentTime);

    this.gl.clear();
    // this.gl.render(this.scene, this.camera);
    this.composer.render();

    this.emit("endFrame", currentTime, this);

    this.gl.info.reset();
  };

  /** Translate a Worldview CameraState to the three.js coordinate system */
  setCameraState(cameraState: CameraState): void {
    this.camera.position
      .setFromSpherical(
        tempSpherical.set(cameraState.distance, cameraState.phi, -cameraState.thetaOffset),
      )
      .applyAxisAngle(UNIT_X, PI_2);
    this.camera.position.add(
      tempVec.set(
        cameraState.targetOffset[0],
        cameraState.targetOffset[1],
        cameraState.targetOffset[2], // always 0 in Worldview CameraListener
      ),
    );
    this.camera.quaternion.setFromEuler(
      tempEuler.set(cameraState.phi, 0, -cameraState.thetaOffset, "ZYX"),
    );
    this.camera.updateProjectionMatrix();
  }

  resizeHandler = (size: THREE.Vector2): void => {
    this.gl.setPixelRatio(window.devicePixelRatio);
    this.gl.setSize(size.width, size.height);
    this.target.setSize(size.width, size.height);
    this.composer.setSize(size.width, size.height);
    for (const pass of this.composer.passes) {
      (pass as Partial<ShaderPass>).uniforms?.["resolution"]?.value.set(
        1 / size.width,
        1 / size.height,
      );
    }
    this.camera.aspect = size.width / size.height;
    this.camera.updateProjectionMatrix();
    const renderSize = this.gl.getDrawingBufferSize(tempVec2);
    log.debug(`Resized renderer to ${renderSize.width}x${renderSize.height}`);
    this.animationFrame();
  };

  clickHandler = (cursorCoords: THREE.Vector2): void => {
    // Clear the outline pass and render the scene again to update this.gl.renderLists
    this.outlinePass.selectedObjects.length = 0;
    this.animationFrame();

    // Render a single pixel using a fragment shader that writes object IDs as
    // colors, then read the value of that single pixel back
    const objectId = this.picker.pick(cursorCoords.x, cursorCoords.y);
    if (objectId < 0) {
      return;
    }

    // Traverse the scene looking for this objectId
    const obj = this.scene.getObjectById(objectId);

    // Find the first ancestor of the clicked object that has a name
    // TODO: We should probably use a better way to identify the clicked object
    let parentObj = obj;
    while (parentObj && parentObj.name === "") {
      parentObj = parentObj.parent ?? undefined;
    }
    if (!parentObj) {
      return;
    }

    // Highlight all of the Mesh components of the clicked object
    const selected: THREE.Object3D[] = [];
    parentObj.traverseVisible((child) => {
      if (
        (child as Partial<THREE.Mesh>).isMesh === true &&
        child.type !== "Line2" &&
        child.type !== "LineSegments"
      ) {
        selected.push(child);
      }
    });
    this.outlinePass.selectedObjects = selected;

    // Re-render with the selected object
    this.animationFrame();
  };
}
