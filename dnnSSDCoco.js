const cv = require('opencv4nodejs');
const fs = require('fs');
const path = require('path');
const classNames = require('./dnnCocoClassNames');


// replace with path where you unzipped inception model
const ssdcocoModelPath = './data/SSD_300x300';

const prototxt = path.resolve(ssdcocoModelPath, 'deploy.prototxt');
const modelFile = path.resolve(ssdcocoModelPath, 'VGG_coco_SSD_300x300_iter_400000.caffemodel');

if (!fs.existsSync(prototxt) || !fs.existsSync(modelFile)) {
  console.log('could not find ssdcoco model');
  console.log('download the model from: https://drive.google.com/file/d/0BzKzrI_SkD1_dUY1Ml9GRTFpUWc/view');
  throw new Error('exiting: could not find ssdcoco model');
}

// initialize ssdcoco model from prototxt and modelFile
const net = cv.readNetFromCaffe(prototxt, modelFile);

const drawRect = (image, rect, color, opts = { thickness: 2 }) =>
  image.drawRectangle(
    rect,
    color,
    opts.thickness,
    cv.LINE_8
  );

const extractResults = function (outputBlob, imgDimensions) {
  return Array(outputBlob.rows).fill(0)
    .map((res, i) => {
      const classLabel = outputBlob.at(i, 1);
      const confidence = outputBlob.at(i, 2);
      const bottomLeft = new cv.Point(
        outputBlob.at(i, 3) * imgDimensions.cols,
        outputBlob.at(i, 6) * imgDimensions.rows
      );
      const topRight = new cv.Point(
        outputBlob.at(i, 5) * imgDimensions.cols,
        outputBlob.at(i, 4) * imgDimensions.rows
      );
      const rect = new cv.Rect(
        bottomLeft.x,
        topRight.y,
        topRight.x - bottomLeft.x,
        bottomLeft.y - topRight.y
      );

      return ({
        classLabel,
        confidence,
        rect
      });
    });
};

function classifyImg(img) {
  // ssdcoco model works with 300 x 300 images
  const imgResized = img.resize(300, 300);

  // network accepts blobs as input
  const inputBlob = cv.blobFromImage(imgResized);
  net.setInput(inputBlob);

  // forward pass input through entire network, will return
  // classification result as 1x1xNxM Mat
  let outputBlob = net.forward();
  // extract NxM Mat
  outputBlob = outputBlob.flattenFloat(outputBlob.sizes[2], outputBlob.sizes[3]);

  return extractResults(outputBlob, img)
    .map(r => Object.assign({}, r, { className: classNames[r.classLabel] }));
}

const makeDrawClassDetections = predictions => (drawImg, className, getColor, thickness = 2) => {
  predictions
    .filter(p => classNames[p.classLabel] === className)
    .forEach(p => drawRect(drawImg, p.rect, getColor(), { thickness }));
  return drawImg;
};

const runDetectDishesExample = () => {
  const img = cv.imread('./data/images/dishes.jpg');
  const minConfidence = 0.2;

  const predictions = classifyImg(img).filter(res => res.confidence > minConfidence);
  predictions.forEach(p => console.log(p));
  console.log();

  const drawClassDetections = makeDrawClassDetections(predictions);

  const classColors = {
    fork: new cv.Vec(0, 255, 0),
    bowl: new cv.Vec(255, 0, 0),
    'wine glass': new cv.Vec(0, 0, 255),
    cup: new cv.Vec(0, 255, 255)
  };

  const legendLeftTop = new cv.Point(580, 20);
  const alpha = 0.4;
  cv.drawTextBox(
    img,
    legendLeftTop,
    Object.keys(classColors).map(className => ({
      text: className,
      fontSize: 0.8,
      color: classColors[className]
    })),
    alpha
  );

  Object.keys(classColors).forEach((className) => {
    const color = classColors[className];
    // draw detections
    drawClassDetections(img, className, () => color);
  });

  cv.imshowWait('img', img);
};

const runDetectPeopleExample = () => {
  const img = cv.imread('./data/images/images11.jpeg');
  const minConfidence = 0.2;

  const predictions = classifyImg(img).filter(res => res.confidence > minConfidence);
  predictions.forEach(p => console.log(p));

  const drawClassDetections = makeDrawClassDetections(predictions);

  const getRandomColor = () => new cv.Vec(Math.random() * 255, Math.random() * 255, 255);

  drawClassDetections(img, 'car', getRandomColor);
  cv.imshowWait('img', img);
};

// runDetectDishesExample();
runDetectPeopleExample();
