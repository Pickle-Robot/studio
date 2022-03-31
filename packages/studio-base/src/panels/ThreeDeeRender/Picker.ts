// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/
//
// Adapted from <https://github.com/bzztbomb/three_js_gpu_picking/blob/main/src/gpupicker.js>
// released under the public domain. Original authors:
// - bzztbomb https://github.com/bzztbomb
// - jfaust https://github.com/jfaust

import * as THREE from "three";

type Camera = THREE.PerspectiveCamera | THREE.OrthographicCamera;

const AlwaysPickObject = (_obj: THREE.Object3D) => true;
// This works around an incorrect method definition, where passing null is valid
const NullScene = ReactNull as unknown as THREE.Scene;

export class Picker {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: Camera;
  private shouldPickObjectCB: (object: THREE.Object3D) => boolean;
  private materialCache = new Map<number, THREE.ShaderMaterial>();
  private emptyScene: THREE.Scene;
  private pixelBuffer: Uint8Array;
  private clearColor = new THREE.Color(0xffffff);
  private currClearColor = new THREE.Color();
  private pickingTarget: THREE.WebGLRenderTarget;

  constructor(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: Camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.shouldPickObjectCB = AlwaysPickObject;

    // This is the 1x1 pixel render target we use to do the picking
    this.pickingTarget = new THREE.WebGLRenderTarget(1, 1, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      encoding: THREE.LinearEncoding,
    });
    this.pixelBuffer = new Uint8Array(4 * this.pickingTarget.width * this.pickingTarget.height);
    // We need to be inside of .render in order to call renderBufferDirect in renderList() so create an empty scene
    // and use the onAfterRender callback to actually render geometry for picking.
    this.emptyScene = new THREE.Scene();
    this.emptyScene.onAfterRender = this.handleAfterRender;
  }

  dispose(): void {
    for (const material of this.materialCache.values()) {
      material.dispose();
    }
    this.materialCache.clear();
    this.pickingTarget.dispose();
  }

  pick(x: number, y: number, shouldPickObject = AlwaysPickObject): number {
    this.shouldPickObjectCB = shouldPickObject;
    const pixelRatio = this.renderer.getPixelRatio();
    const xs = x * pixelRatio;
    const ys = y * pixelRatio;
    const w = this.renderer.domElement.width;
    const h = this.renderer.domElement.height;
    // Set the projection matrix to only look at the pixel we are interested in
    this.camera.setViewOffset(w, h, xs, ys, 1, 1);

    const currRenderTarget = this.renderer.getRenderTarget();
    const currAlpha = this.renderer.getClearAlpha();
    this.renderer.getClearColor(this.currClearColor);
    this.renderer.setRenderTarget(this.pickingTarget);
    this.renderer.setClearColor(this.clearColor);
    this.renderer.setClearAlpha(1);
    this.renderer.clear();
    this.renderer.render(this.emptyScene, this.camera);
    this.renderer.readRenderTargetPixels(
      this.pickingTarget,
      0,
      0,
      this.pickingTarget.width,
      this.pickingTarget.height,
      this.pixelBuffer,
    );
    this.renderer.setRenderTarget(currRenderTarget);
    this.renderer.setClearColor(this.currClearColor, currAlpha);
    this.camera.clearViewOffset();

    const val =
      (this.pixelBuffer[0]! << 24) +
      (this.pixelBuffer[1]! << 16) +
      (this.pixelBuffer[2]! << 8) +
      this.pixelBuffer[3]!;
    return val;
  }

  private handleAfterRender = (): void => {
    // This is the magic, these render lists are still filled with valid data.
    // So we can submit them again for picking and save lots of work!
    const renderList = this.renderer.renderLists.get(this.scene, 0);
    renderList.opaque.forEach(this.processItem);
    renderList.transmissive.forEach(this.processItem);
    renderList.transparent.forEach(this.processItem);
  };

  private processItem = (renderItem: THREE.RenderItem): void => {
    const object = renderItem.object;
    const objId = object.id;
    const material = renderItem.material;
    const geometry = renderItem.geometry;
    if (
      !geometry || // Skip if geometry is not defined
      material.type === "LineBasicMaterial" || // Skip marker outlines
      material.type === "LineMaterial" || // TODO(jhurliman): Render this, requires a custom fragment shader
      !this.shouldPickObjectCB(object) // Skip if user callback returns false
    ) {
      return;
    }

    const useInstancing = (object as Partial<THREE.InstancedMesh>).isInstancedMesh === true ? 1 : 0;
    const frontSide = material.side === THREE.FrontSide ? 1 : 0;
    const doubleSide = material.side === THREE.DoubleSide ? 1 : 0;
    const sprite = material.type === "SpriteMaterial" ? 1 : 0;
    const sizeAttenuation =
      (material as Partial<THREE.PointsMaterial>).sizeAttenuation === true ? 1 : 0;
    const index =
      (useInstancing << 0) |
      (frontSide << 1) |
      (doubleSide << 2) |
      (sprite << 3) |
      (sizeAttenuation << 4);
    const pickingMaterial = renderItem.object.userData.pickingMaterial as
      | THREE.ShaderMaterial
      | undefined;
    let renderMaterial = pickingMaterial ?? this.materialCache.get(index);
    if (!renderMaterial) {
      let vertexShader = THREE.ShaderChunk.meshbasic_vert;
      if (sprite === 1) {
        vertexShader = THREE.ShaderChunk.sprite_vert!;
      }
      if (sizeAttenuation === 1) {
        vertexShader = "#define USE_SIZEATTENUATION\n\n" + vertexShader;
      }
      renderMaterial = new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader: `
           uniform vec4 objectId;
           void main() {
             gl_FragColor = objectId;
           }
         `,
        side: material.side,
      });
      renderMaterial.uniforms = { objectId: { value: [1.0, 1.0, 1.0, 1.0] } };
      this.materialCache.set(index, renderMaterial);
    }
    if (sprite === 1) {
      renderMaterial.uniforms.rotation = { value: (material as THREE.SpriteMaterial).rotation };
      renderMaterial.uniforms.center = { value: (object as THREE.Sprite).center };
    }
    renderMaterial.uniforms.objectId!.value = [
      ((objId >> 24) & 255) / 255,
      ((objId >> 16) & 255) / 255,
      ((objId >> 8) & 255) / 255,
      (objId & 255) / 255,
    ];
    renderMaterial.uniformsNeedUpdate = true;
    this.renderer.renderBufferDirect(
      this.camera,
      NullScene,
      geometry,
      renderMaterial,
      object,
      ReactNull,
    );
  };
}
