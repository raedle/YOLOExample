import { Alert } from "react-native";
import { Image, media, Module, Tensor, torch, torchvision } from "react-native-pytorch-core";

import CocoNames from './CocoClasses.json';

// Original source: https://github.com/pytorch/android-demo-app/blob/master/ObjectDetection/app/src/main/java/org/pytorch/demo/objectdetection/PrePostProcessor.java
// The code was adjusted to match PyTorch Live API

// The image size is defined by the expected model input.
const IMAGE_SIZE = 640;

// Helper type to store left, top, right, bottom bounds
type Rect = [number, number, number, number];

type BoundingBox = {
  // The detected object label
  label: string,
  // The confidence score
  score: number,
  // The object bounds
  rect: Rect,
}

/**
 * Detect objects in an image. The model needs to be a PyTorch model loaded in
 * the lite interpreter runtime and be compatible with the implemented
 * preprocessing and postprocessing steps.
 *
 * @param model Model loaded for lite interpreter runtime.
 * @param image Image object either from the camera or loaded via url or
 * bundle.
 * @returns Detected objects with their score, label, and bounds (left, top,
 * right, bottom).
 */
export async function detectObjects(model: Module, image: Image) {
  // BEGIN: Capture performance measure for preprocessing
  const startPackTime = performance.now();
  const height = image.getHeight();
  const width = image.getWidth();
  // Convert camera image to blob (raw image data in HWC format)
  const blob = media.toBlob(image);
  // Get tensor from blob and define HWC shape for tensor
  let tensor = torch.fromBlob(blob, [height, width, 3]);
  // Change tensor shape from HWC to CHW (channel first) (3, H, C)
  tensor = tensor.permute([2, 0, 1]);
  // Convert to float tensor and values to [0, 1]
  tensor = tensor.div(255);
  // Resize image tensor to match model input shape (3, 640, 640)
  const resize = torchvision.transforms.resize([IMAGE_SIZE, IMAGE_SIZE]);
  tensor = resize(tensor);
  // Center crop image tensor (in some cases the resize leads to +/- 1 size
  // difference, e.g., (3, 640, 641) which will fail inference)
  const centerCrop = torchvision.transforms.centerCrop([IMAGE_SIZE]);
  tensor = centerCrop(tensor);
  // Add dimension for batch size (1, 3, 640, 640)
  tensor = tensor.unsqueeze(0);
  // END: Capture performance measure for preprocessing
  const packTime = global.performance.now() - startPackTime;

  try {
    // BEGIN: Capture performance measure for inference
    const startInferencTime = global.performance.now();
    // Run ML inference
    const output = await model.forward<Tensor, Tensor[]>(tensor);
    // END: Capture performance measure for inference
    const inferenceTime = global.performance.now() - startInferencTime;

    // BEGIN: Capture performance measure for postprocessing
    const startUnpackTime = global.performance.now();
    // Note: The toTensor API is likely going to change
    const prediction = output[0];
    // Get image width/height to adjust bounds returned by model to image size
    const imageWidth = image.getWidth();
    const imageHeight = image.getHeight();
    const imgScaleX = imageWidth / IMAGE_SIZE;
    const imgScaleY = imageHeight / IMAGE_SIZE;
    // Get label, score, and bounds from model inference result
    const results = outputsToNMSPredictions(
      prediction[0],
      imgScaleX,
      imgScaleY,
    );

    // END: Capture performance measure for postprocessing
    const unpackTime = global.performance.now() - startUnpackTime;

    console.log(`pack time ${packTime.toFixed(3)} ms`);
    console.log(`inference time ${inferenceTime.toFixed(3)} ms`);
    console.log(`unpack time ${unpackTime.toFixed(3)} ms`);

    return results;
  }
  catch (error: any) {
    Alert.alert('Error', error);
  }
  return [];
}

/**
 * Bounding boxes of detected objects with their label and score. The function
 * filters detections based on a probability threshold. It limits the to a max
 * of 15 detected objects.
 *
 * The function also performs a NMS and IOU to filter out overlapping bounding
 * boxes.
 *
 * @param prediction Predictions from the model.
 * @param imgScaleX Scale x-bounds to match input image size
 * @param imgScaleY Scale y-bounds to match input image size
 * @returns Detected objects with label, score, and bounds
 */
function outputsToNMSPredictions(
  prediction: Tensor,
  imgScaleX: number,
  imgScaleY: number,
) {
  const threshold = 0.3;
  const limit = 15;
  const results = [];
  // Get number of rows (decided by YOLO model)
  const rows = prediction.shape[0];
  // The first five items are bounds (x, y, w, h) and score. The remaining
  // items are probabilities for each of the classes (likely 80 Coco classes).
  const nc = prediction.shape[1] - 5;
  for (let i = 0; i < rows; i++) {
    // Access tensor data
    // Note: The data API does not exist in PyTorch Python and can change
    const outputs = prediction[i].data();
    // Filter detections lower than the thresold
    if (outputs[4] > threshold) {
      // Get object bounds
      const x = outputs[0];
      const y = outputs[1];
      const w = outputs[2];
      const h = outputs[3];

      // Scale bounds to input image size
      const left = imgScaleX * (x - w / 2);
      const top = imgScaleY * (y - h / 2);
      const right = imgScaleX * (x + w / 2);
      const bottom = imgScaleY * (y + h / 2);

      // Get top class label (could be done by slicing the data from 5 to nc + 5
      // and then argmax)
      let max = outputs[5];
      let cls = 0;
      for (let j = 0; j < nc; j++) {
        if (outputs[j + 5] > max) {
          max = outputs[j + 5];
          cls = j;
        }
      }

      // Object bounds adjusted to input image
      const rect: Rect = [
        left,
        top,
        right,
        bottom,
      ];

      // Object label based on Coco classes
      const label = CocoNames[cls];

      // Put together result object
      const result = {
        label,
        score: outputs[4],
        rect,
      };
      results.push(result);
    }
  }
  return nonMaxSuppression(results, limit, threshold);
}

/**
 * Select one object out of many overlapping objects.
 *
 * @param boxes Detected objects over a certain probability threshold
 * @param limit Maximum number of objects in final result
 * @param threshold The IOU threshold
 * @returns Detected objects with label, score, and bounds
 */
function nonMaxSuppression(
  boxes: BoundingBox[],
  limit: number,
  threshold: number,
) {
  // Do an argsort on the confidence scores, from high to low.
  const newBoxes = boxes.sort((a, b) => {
    return a.score - b.score;
  });

  const selected = [];
  const active = new Array(newBoxes.length).fill(true);
  let numActive = active.length;

  // The algorithm is simple: Start with the box that has the highest score.
  // Remove any remaining boxes that overlap it more than the given threshold
  // amount. If there are any boxes left (i.e. these did not overlap with any
  // previous boxes), then repeat this procedure, until no more boxes remain
  // or the limit has been reached.
  let done = false;
  for (let i = 0; i < newBoxes.length && !done; i++) {
    if (active[i]) {
      const boxA = newBoxes[i];
      selected.push(boxA);
      if (selected.length >= limit) break;

      for (let j = i + 1; j < newBoxes.length; j++) {
        if (active[j]) {
          const boxB = newBoxes[j];
          if (IOU(boxA.rect, boxB.rect) > threshold) {
            active[j] = false;
            numActive -= 1;
            if (numActive <= 0) {
              done = true;
              break;
            }
          }
        }
      }
    }
  }
  return selected;
}

/**
 * Computes intersection-over-union overlap between two bounding boxes.
 *
 * @param rect1 Bounds of object 1
 * @param rect2 Bounds of object 2
 * @returns The IOU value
 */
function IOU(rect1: Rect, rect2: Rect) {
  let areaA = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]);
  if (areaA <= 0.0) return 0.0;

  let areaB = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]);
  if (areaB <= 0.0) return 0.0;

  const intersectionMinX = Math.max(rect1[0], rect2[0]);
  const intersectionMinY = Math.max(rect1[1], rect2[1]);
  const intersectionMaxX = Math.min(rect1[2], rect2[2]);
  const intersectionMaxY = Math.min(rect1[3], rect2[3]);
  const intersectionArea =
    Math.max(intersectionMaxY - intersectionMinY, 0) *
    Math.max(intersectionMaxX - intersectionMinX, 0);
  return intersectionArea / (areaA + areaB - intersectionArea);
}
