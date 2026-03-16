import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  // Phase 1: Grayscale
  private toGrayscale(imageData: ImageData): Float32Array {
    const data = imageData.data;
    const gray = new Float32Array(data.length / 4);
    for (let i = 0; i < data.length; i += 4) {
      gray[i / 4] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    return gray;
  }

  // Phase 2: Gaussian blur
  private gaussianBlur(gray: Float32Array, width: number, height: number): Float32Array {
    const kernel = [[1,2,1],[2,4,2],[1,2,1]];
    const result = new Float32Array(gray.length);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++)
          for (let kx = -1; kx <= 1; kx++)
            sum += gray[(y + ky) * width + (x + kx)] * kernel[ky + 1][kx + 1];
        result[y * width + x] = sum / 16;
      }
    }
    return result;
  }

  // Phase 3: Otsu automatic threshold
  // Source: https://gist.github.com/zz85/2ebc8e4da705dc3244200de564ab5557
  private otsuThreshold(gray: Float32Array): number {
    const hist = new Array(256).fill(0);
    for (let i = 0; i < gray.length; i++) hist[Math.floor(gray[i])]++;
    const total = gray.length;
    let sum = 0;
    for (let t = 0; t < 256; t++) sum += t * hist[t];
    let sumB = 0, wB = 0, wF = 0, varMax = 0, threshold = 0;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      wF = total - wB;
      if (wF === 0) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const v = wB * wF * (mB - mF) * (mB - mF);
      if (v > varMax) { varMax = v; threshold = t; }
    }
    return threshold;
  }

  // Phase 4: Binary threshold — shapes=1, background=0
  private binarize(gray: Float32Array, threshold: number, shapeIsDark: boolean): Uint8Array {
    const binary = new Uint8Array(gray.length);
    for (let i = 0; i < gray.length; i++) {
      binary[i] = shapeIsDark
        ? (gray[i] < threshold ? 1 : 0)
        : (gray[i] >= threshold ? 1 : 0);
    }
    return binary;
  }

  // Phase 5: Morphological closing with radius 1
  // Fills small gaps without rounding corners
  private morphClose(binary: Uint8Array, width: number, height: number): Uint8Array {
    const dilated = new Uint8Array(binary.length);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let hasShape = false;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++)
            if (binary[(y+dy)*width+(x+dx)] === 1) { hasShape = true; break; }
        dilated[y * width + x] = hasShape ? 1 : 0;
      }
    }
    const result = new Uint8Array(binary.length);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let allShape = true;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++)
            if (dilated[(y+dy)*width+(x+dx)] === 0) { allShape = false; break; }
        result[y * width + x] = allShape ? 1 : 0;
      }
    }
    return result;
  }

  // Phase 6: Find connected regions on binary image using BFS
  // Key insight: run on binary image not Canny edges
  // Source: https://pyimagesearch.com/2016/02/08/opencv-shape-detection/
  private findContoursBinary(binary: Uint8Array, width: number, height: number): {x: number, y: number}[][] {
    const visited = new Uint8Array(binary.length);
    const contours: {x: number, y: number}[][] = [];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (binary[idx] === 1 && !visited[idx]) {
          const region: {x: number, y: number}[] = [];
          const queue = [{x, y}];
          visited[idx] = 1;
          while (queue.length > 0) {
            const p = queue.shift()!;
            region.push(p);
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                const nx = p.x + dx, ny = p.y + dy;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                const ni = ny * width + nx;
                if (binary[ni] === 1 && !visited[ni]) {
                  visited[ni] = 1;
                  queue.push({x: nx, y: ny});
                }
              }
            }
          }
          if (region.length > 100) contours.push(region);
        }
      }
    }
    return contours;
  }

  // Phase 7: Extract boundary pixels from filled region
  private extractBoundary(
    region: {x: number, y: number}[],
    binary: Uint8Array,
    width: number,
    height: number
  ): {x: number, y: number}[] {
    const boundary: {x: number, y: number}[] = [];
    for (const p of region) {
      let isBoundary = false;
      for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const nx = p.x + dx, ny = p.y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height || binary[ny * width + nx] === 0) {
          isBoundary = true;
          break;
        }
      }
      if (isBoundary) boundary.push(p);
    }
    return boundary;
  }

  // Phase 8: Bounding box
  private getBoundingBox(points: {x: number, y: number}[]) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    return {
      x: minX, y: minY,
      width: maxX - minX,
      height: maxY - minY,
      centerX: Math.round((minX + maxX) / 2),
      centerY: Math.round((minY + maxY) / 2),
    };
  }

  // Phase 9: Sort boundary by angle around center
  private sortByAngle(
    pts: {x: number, y: number}[],
    cx: number, cy: number
  ): {x: number, y: number}[] {
    return [...pts].sort((a, b) =>
      Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
    );
  }

  // Phase 10: Douglas-Peucker polygon approximation
  // epsilon = 2% of perimeter
  // Source: https://pyimagesearch.com/2016/02/08/opencv-shape-detection/
  private douglasPeucker(points: {x:number,y:number}[], epsilon: number): {x:number,y:number}[] {
    if (points.length < 3) return points;
    let maxDist = 0, maxIdx = 0;
    const start = points[0], end = points[points.length - 1];
    for (let i = 1; i < points.length - 1; i++) {
      const dx = end.x - start.x, dy = end.y - start.y;
      const len = Math.sqrt(dx*dx + dy*dy);
      const dist = len === 0 ? 0
        : Math.abs(dy*points[i].x - dx*points[i].y + end.x*start.y - end.y*start.x) / len;
      if (dist > maxDist) { maxDist = dist; maxIdx = i; }
    }
    if (maxDist > epsilon) {
      const left  = this.douglasPeucker(points.slice(0, maxIdx + 1), epsilon);
      const right = this.douglasPeucker(points.slice(maxIdx), epsilon);
      return [...left.slice(0, -1), ...right];
    }
    return [start, end];
  }

  // Phase 11: Classify by circularity + vertex count + area
  // circ > 0.90: very round — vertex count distinguishes circle/pentagon/rotated-rect
  // circ 0.60-0.90: medium — vertex count reliable, area catches tiny triangles
  //   (small shapes can't be cleanly approximated by Douglas-Peucker)
  // circ < 0.60: irregular — star or triangle
  // Source: https://pyimagesearch.com/2016/02/08/opencv-shape-detection/
  private classifyShape(vertices: number, circularity: number, area: number): DetectedShape["type"] {
    if (circularity > 0.90) {
      if (vertices >= 9) return "circle";
      if (vertices >= 7) return "pentagon";
      return "rectangle";
    }
    if (circularity > 0.60) {
      // small shapes with medium circularity = tiny triangle
      // (blur prevents clean 3-vertex approximation on small shapes)
      if (area < 1000 && vertices <= 6) return "triangle";
      if (vertices <= 5) return "triangle";
      return "rectangle";
    }
    if (vertices >= 10) return "star";
    if (vertices <= 4) return "triangle";
    return "pentagon";
  }

  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const shapes: DetectedShape[] = [];
    const { width, height } = imageData;

    const gray    = this.toGrayscale(imageData);
    const blurred = this.gaussianBlur(gray, width, height);

    const threshold   = this.otsuThreshold(blurred);
    const darkCount   = Array.from(blurred).filter(v => v < threshold).length;
    const shapeIsDark = darkCount < blurred.length * 0.5;
    console.log(`Otsu threshold=${threshold} shapeIsDark=${shapeIsDark}`);

    const binary  = this.binarize(blurred, threshold, shapeIsDark);
    const closed  = this.morphClose(binary, width, height);
    const regions = this.findContoursBinary(closed, width, height);
    console.log(`found ${regions.length} regions`);

    for (const region of regions) {
      const bbox = this.getBoundingBox(region);
      if (bbox.width < 15 || bbox.height < 15) continue;
      if (bbox.width > width * 0.98 || bbox.height > height * 0.98) continue;

      const area = region.length;
      if (area < 200) continue;

      const boundary = this.extractBoundary(region, closed, width, height);
      if (boundary.length < 20) continue;

      const perimeter   = boundary.length;
      const circularity = Math.min(1.0, (4 * Math.PI * area) / (perimeter * perimeter));

      const sorted     = this.sortByAngle(boundary, bbox.centerX, bbox.centerY);
      const simplified = this.douglasPeucker(sorted, perimeter * 0.02);
      const vertices   = simplified.length;

      console.log(`region: vertices=${vertices} circ=${circularity.toFixed(3)} area=${area} perim=${perimeter} w=${bbox.width} h=${bbox.height}`);

      // pass area for small shape detection
      const type = this.classifyShape(vertices, circularity, area);

      shapes.push({
        type,
        confidence: circularity > 0.85 ? 0.95 : 0.80,
        boundingBox: { x: bbox.x, y: bbox.y, width: bbox.width, height: bbox.height },
        center: { x: bbox.centerX, y: bbox.centerY },
        area
      });
    }

    const processingTime = performance.now() - startTime;
    return { shapes, processingTime, imageWidth: width, imageHeight: height };
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        resolve(this.ctx.getImageData(0, 0, img.width, img.height));
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById("originalCanvas") as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);
    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById("testImages") as HTMLDivElement;
    this.evaluateButton = document.getElementById("evaluateButton") as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById("evaluationResults") as HTMLDivElement;
    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(this.detector, this.evaluateButton, this.evaluationResultsDiv);
    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) await this.processImage(file);
    });
    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";
      const imageData = await this.detector.loadImage(file);
      const results   = await this.detector.detectShapes(imageData);
      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;
    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;
    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${shape.type.charAt(0).toUpperCase() + shape.type.slice(1)}</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(1)})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html += "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }
    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module     = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();
      let html = '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;
      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName.replace(/[_-]/g, " ").replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}"
               onclick="loadTestImage('${imageName}', '${dataUrl}')"
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });
      html += "</div>";
      this.testImagesDiv.innerHTML = html;
      this.selectionManager.setupSelectionControls();
      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response  = await fetch(dataUrl);
          const blob      = await response.blob();
          const file      = new File([blob], name, { type: "image/svg+xml" });
          const imageData = await this.detector.loadImage(file);
          const results   = await this.detector.detectShapes(imageData);
          this.displayResults(results);
          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };
      (window as any).toggleImageSelection = (event: MouseEvent, imageName: string) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };
      (window as any).triggerFileUpload = () => { this.imageInput.click(); };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => { new ShapeDetectionApp(); });