/*jshint esversion: 9 */


let colorData;

let nn;
let xs;
let ys;

let pLoss, pLabel, pRed, pGreen, pBlue;

let rSlider, gSlider, bSlider;

let labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
];

function preload() {
  colorData = loadJSON('colorData.json');
}

function setup() {
  createCanvas(400, 400);
  pLabel = createP('Label');
  pLoss = createP('loss');
  pRed = createP('Red: 255');
  rSlider = createSlider(0, 255, 255, 1);
  pGreen = createP('Green: 255');
  gSlider = createSlider(0, 255, 255, 1);
  pBlue = createP('Blue: 0');
  bSlider = createSlider(0, 255, 0, 1);

  let colors = [];
  let labels = [];
  for (let record of colorData.entries) {
    //normalize the color values between 0 and 1
    let col = [record.r / 255, record.g / 255, record.b / 255];
    colors.push(col);
    labels.push(labelList.indexOf(record.label));
  }

  xs = tf.tensor2d(colors);
  let labelsTensor = tf.tensor1d(labels, 'int32');
  ys = tf.oneHot(labelsTensor, 9);
  labelsTensor.dispose();

  //create the neural network model
  nn = tf.sequential();

  //create the layers
  let hidden = tf.layers.dense({
    inputShape: 3,
    units: 32,
    activation: 'sigmoid'
  });

  let output = tf.layers.dense({
    units: 9,
    activation: 'softmax'
  });

  //add the layers to the nn model
  nn.add(hidden);
  nn.add(output);

  //create an optimizer
  //stochastic gradient descent
  let learningRate = 0.2;
  let optimizer = tf.train.adam(learningRate);

  //compile the nn model
  nn.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy'
  });

  train().then((response) => {
    print(response.history.loss);
  });

}

async function train() {
  //put the data in the model
  const fitConfig = {
    epochs: 100,
    shuffle: true,
    validationSplit: 0.1,
    callbacks: {
      onTrainBegin: () => print('Training Start'),
      onTrainEnd: () => print('Training Complete'),
      onBatchEnd: tf.nextFrame,
      onEpochEnd: (num, logs) => {
        //at the end of each epoch, print the loss
        print('Epoch: ' + num);
        //pLoss.html('Loss: ' + logs.val_loss);
      }

    }
  }
  return await nn.fit(xs, ys, fitConfig);
}

function draw() {
  // background(0);
  // stroke(255);
  // strokeWeight(4);
  // line(frameCount % width, 0, frameCount % width, height);

  let r = rSlider.value();
  let g = gSlider.value();
  let b = bSlider.value();
  
  pRed.html("Red: " + r);
  pGreen.html("Green: " + g);
  pBlue.html("Blue:" + b);

  background(r, g, b);

  tf.tidy(() => {
    const xs = tf.tensor2d([
      [r / 255, g / 255, b / 255]
    ]);

    let results = nn.predict(xs);
    let index = results.argMax(1).dataSync();
    let label = labelList[index];
    let confidence = results.dataSync()[index];
    pLoss.html("Confidence: " + round(100 *confidence, 0) + "%");
    pLabel.html(label);
  });

}