const cv = require('opencv4nodejs');
const fs = require('fs');
const path = require('path');
const mongoose = require('mongoose');
const fetch = require('node-fetch');
const classNames = require('./dnnCocoClassNames');

// replace with path where you unzipped inception model
const ssdcocoModelPath = './data/SSD_300x300';
const Schema = mongoose.Schema;
const cameraDataSchema = new Schema({
  CameraID: {
    type: String,
    required: true
  },
  label: String,
  name: String,
  confidence: Number,
  datetime: Date,
  location: {
    type: {
      type: String,
      enum: ['Point'],
      required: true
    },
    coordinates: {
      type: [Number],
      required: true
    }
  }
}, { timestamps: true });

// var dataSchema = new Schema({
 
//   label: String,
//   name: String,
//   confidence: Number
// });

const results = {
  "odata.metadata": "http://datamall2.mytransport.sg/ltaodataservice/$metadata#CameraImageSet",
  "value": [
      {
          "CameraID": "1001",
          "Latitude": 1.29531332,
          "Longitude": 103.871146,
          "ImageLink": "https://s3-ap-southeast-1.amazonaws.com/mtpdm/2019-07-22/19-41/1001_1928_20190722193607_8fe4eb.jpg"
      },
      {
          "CameraID": "1002",
          "Latitude": 1.319541067,
          "Longitude": 103.8785627,
          "ImageLink": "https://s3-ap-southeast-1.amazonaws.com/mtpdm/2019-07-22/19-41/1002_1928_20190722193600_4e1385.jpg"
      },
      {
          "CameraID": "9706",
          "Latitude": 1.41270056,
          "Longitude": 103.80642712,
          "ImageLink": "https://s3-ap-southeast-1.amazonaws.com/mtpdm/2019-07-22/19-41/9706_1936_20190722193900_033984.jpg"
      }
  ]
}
const CameraData = mongoose.model('Database', cameraDataSchema);
// var dataDetails = mongoose.model('datavalues', dataSchema);
/**
 * Connect to MongoDB.
 */
const MONGODB_URI = 'mongodb://localhost:27017/testhack';
mongoose.set('useFindAndModify', false);
mongoose.set('useCreateIndex', true);
mongoose.set('useNewUrlParser', true);
mongoose.connect(MONGODB_URI);
mongoose.connection.on('error', (err) => {
  console.error(err);
  console.log('%s MongoDB connection error. Please make sure MongoDB is running.');
  process.exit();
});


const prototxt = path.resolve(ssdcocoModelPath, 'deploy.prototxt');
const modelFile = path.resolve(ssdcocoModelPath, 'VGG_coco_SSD_300x300_iter_400000.caffemodel');

if (!fs.existsSync(prototxt) || !fs.existsSync(modelFile)) {
  console.log('could not find ssdcoco model');
  console.log('download the model from: https://drive.google.com/file/d/0BzKzrI_SkD1_dUY1Ml9GRTFpUWc/view');
  throw new Error('exiting: could not find ssdcoco model');
}
const fetchData = async (url, parameters = {}) => {
// let results = null;
//   await fetch('http://datamall2.mytransport.sg/ltaodataservice/Traffic-Images', {
//     method: 'GET',
//     // body:    JSON.stringify(body),
//     headers: { 'Content-Type': 'application/json', AccountKey: 'kzlesW8dS92tEC31eTEYVw==' },
//   })
//     .then(res => res.text())
//     .then(body => {
//       results = body;
//     });
    // return results;
    return results;
  // return {results: ''};
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

const runDetectPeopleExample = async () => {
  const allDatas = await fetchData();
  // console.log('allDatas');
  // console.log(allDatas);
  console.log(allDatas);
//   const datain = {
//     "CameraID": "8704",
//     "Latitude": 1.3899,
//     "Longitude": 103.74843,
//     "ImageLink": "https://s3-ap-southeast-1.amazonaws.com/mtpdm/2019-07-22/17-48/8704_1744_20190722174503_6070b9.jpg"
// };
const datain = allDatas.value[0];

  const imageUrl = 'https://s3-ap-southeast-1.amazonaws.com/mtpdm/2019-07-22/17-48/8702_1744_20190722174506_27f879.jpg';
  // const fileName = imageUrl.substring(imageUrl.lastIndexOf('/')+1);  
  const fileName = imageUrl.split('/').reverse()[0];
// console.log(imageUrl.split('/').reverse()[1]);
// console.log(imageUrl.split('/').reverse()[2]);

const dateValue = imageUrl.split('/').reverse()[1].split('-');
const dateTimeValue = imageUrl.split('/').reverse()[2].split('-');

// console.log(dateValue);
// console.log(dateTimeValue);
var currentDateTime = new Date(dateTimeValue[0], dateTimeValue[1], dateTimeValue[2], dateValue[0], dateValue[1]);
// console.log(currentDateTime);

  const fileSavePath = `./fetchimages/${fileName}`;
  // const res = await fetch(imageUrl);
  // await new Promise((resolve, reject) => {
  //   const fileStream = fs.createWriteStream(fileSavePath);
  //   res.body.pipe(fileStream);
  //   res.body.on("error", (err) => {
  //     reject(err);
  //   });
  //   fileStream.on("finish", function() {
  //     resolve();
  //   });
  // });  
//   await fetch(imageUrl)
//   .then(res => {
//     // console.log(res);
// console.log('2');
//       const dest = fs.createWriteStream(fileName);
//       res.body.pipe(dest);
//   });  
  // let image;
  // await fetch('https://s3-ap-southeast-1.amazonaws.com/mtpdm/2019-07-22/17-48/9703_1744_20190722174503_e8f15d.jpg')
  // .then(res => image = res.buffer());

// console.log('1', fileName);
  const img = await cv.imread(fileSavePath);
  // console.log('11 222233');
  const minConfidence = 0.4;
  // console.log('11 2');
  const predictions = await classifyImg(img).filter(res => res.confidence > minConfidence);
  // console.log('11 3');
  predictions.forEach(async (p) => {
    console.log(p);
    const cameraGeo = { type: 'Point', coordinates: [datain.Longitude, datain.Latitude] };
    const datasave = new CameraData({
      CameraID: datain.CameraID,
      location: cameraGeo,
      label: p.classLabel,
      name: p.className,
      confidence: p.confidence.toFixed(2),
      datetime: currentDateTime,
    });
    await datasave.save();
    // const data = new dataDetails(
    //   {
    //     label: p.classLabel,
    //     name: p.className,
    //     confidence: p.confidence.toFixed(2),
    //     datetime: currentDateTime,
    //   });
    // await data.save();
  });
  // console.log('11 4');
  // await fs.unlinkSync(fileSavePath);
  // console.log('11 5');

  // const drawClassDetections = makeDrawClassDetections(predictions);

  // const getRandomColor = () => new cv.Vec(Math.random() * 255, Math.random() * 255, 255);

  // drawClassDetections(img, 'car', getRandomColor);
  // cv.imshowWait('img', img);
};

// runDetectDishesExample();
runDetectPeopleExample();

